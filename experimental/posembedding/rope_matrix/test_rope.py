import os
import numpy as np 
os.environ["COMBINED_ENABLE"] = "1"  # 
os.environ["INF_NAN_MODE_ENABLE"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

import torch
g_ascend_env = 0
try:
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = False

    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False 

    from torch_npu.contrib import transfer_to_npu
    print("import torch_npu success, training in ascend")
    g_ascend_env = 1
except:
    torch_npu = None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("import torch_npu failed, training in gpu")

torch.manual_seed(0)
torch.set_printoptions(precision=16)

import torch.nn as nn
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.profiler import profile, schedule
from torch.autograd.profiler import record_function
from torch.cuda.amp import autocast
import torch.nn.functional as F
from einops import rearrange, repeat


def get_interleave_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16, device='npu')
    for i in range(0,n,2):
        matrix[i+0, i+1] = 1
        matrix[i+1, i+0] = -1
    return matrix


def get_half_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16)
    half = n // 2
    matrix[:half, half:] = torch.eye(half)
    matrix[half:, :half] = -torch.eye(half)
    return matrix.to('npu')


def compose_3matrix(A, B, C):
    # assert A.dim() == B.dim() == C.dim() == 2

    total_rows = A.size(0) + B.size(0) + C.size(0)
    total_cols = A.size(1) + B.size(1) + C.size(1)

    result = torch.zeros((total_rows, total_cols), dtype=torch.bfloat16)

    result[:A.size(0), :A.size(1)] = A

    b_row_start = A.size(0)
    b_col_start = A.size(1)
    result[b_row_start:b_row_start + B.size(0),
           b_col_start:b_col_start + B.size(1)] = B
    
    c_row_start = A.size(0) + B.size(0)
    c_col_start = A.size(1) + B.size(1)
    result[c_row_start:c_row_start + C.size(0),
           c_col_start:c_col_start + C.size(1)] = C
    
    return result.to('npu')


# half
def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# interleave
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, mode: str) -> torch.Tensor:
    sin = sin.unsqueeze(0).unsqueeze(1)
    cos = cos.unsqueeze(0).unsqueeze(1)
    if mode == 'interleave':
        return (tensor * cos) + (rotate_every_two(tensor) * sin)
    elif mode == 'half':
        return (tensor * cos) + (rotate_half(tensor) * sin)
    else:
        raise NotImplementedError("mode error, only support half or interleave")
    

def apply_3drotary_pos_v1(q, k, freqs_cis, mode):
    '''
    小算子实现
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
    k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
    q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
    k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()

    q1 = apply_rotary_pos_emb(q1, sin_h, cos_h, mode)
    k1 = apply_rotary_pos_emb(k1, sin_h, cos_h, mode)
    q2 = apply_rotary_pos_emb(q2, sin_w, cos_w, mode)
    k2 = apply_rotary_pos_emb(k2, sin_w, cos_w, mode)
    q3 = apply_rotary_pos_emb(q3, sin_t, cos_t, mode)
    k3 = apply_rotary_pos_emb(k3, sin_t, cos_t, mode)

    q = torch.concat([q1, q2, q3], dim=-1)
    k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


def apply_3drotary_pos_v2(q, k, freqs_cis, mat1, mat2, mat3):
    '''
    rope_matrix, matrix不合一的实现
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis
    
    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
    k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
    q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
    k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()

    q1 = q1 * cos_h + (q1 @ mat1) * sin_h
    k1 = k1 * cos_h + (k1 @ mat1) * sin_h
    q2 = q2 * cos_w + (q2 @ mat2) * sin_w
    k2 = k2 * cos_w + (k2 @ mat2) * sin_w
    q3 = q3 * cos_t + (q3 @ mat3) * sin_t
    k3 = k3 * cos_t + (k3 @ mat3) * sin_t

    q = torch.concat([q1, q2, q3], dim=-1)
    k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


def apply_3drotary_pos_v3(q, k, freqs_cis, mat, debug=None, high_precision=None):
    '''
    本方案:rope_matrix, matrix合一
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
    cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)

    if debug:
        print(f"q @ mat = {q[0][0][0][:10]}")
    
    if high_precision:
        q = q.float() * cos.float() + (q @ mat).float() * sin.float()
        k = k.float() * cos.float() + (k @ mat).float() * sin.float()
        q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    else:
        q = q * cos + (q @ mat) * sin
        k = k * cos + (k @ mat) * sin
    
    return q, k


def apply_3drotary_pos_v4(q, k, freqs_cis, mode):
    '''
    当前库上融合算子实现
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t
    
    if mode == 'interleave':
        sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
        cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        cos = cos.unsqueeze(0).unsqueeze(1)

        q = torch_npu.npu_rotary_mul(q, cos, sin, mode)
        k = torch_npu.npu_rotary_mul(k, cos, sin, mode)
    elif mode == 'half':
        q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
        k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
        q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
        k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()
        sin_h = sin_h.unsqueeze(0).unsqueeze(1)
        cos_h = cos_h.unsqueeze(0).unsqueeze(1)
        sin_w = sin_w.unsqueeze(0).unsqueeze(1)
        cos_w = cos_w.unsqueeze(0).unsqueeze(1)
        sin_t = sin_t.unsqueeze(0).unsqueeze(1)
        cos_t = cos_t.unsqueeze(0).unsqueeze(1)

        q1 = torch_npu.npu_rotary_mul(q1, cos_h, sin_h, mode)
        k1 = torch_npu.npu_rotary_mul(k1, cos_h, sin_h, mode)
        q2 = torch_npu.npu_rotary_mul(q2, cos_w, sin_w, mode)
        k2 = torch_npu.npu_rotary_mul(k2, cos_w, sin_w, mode)
        q3 = torch_npu.npu_rotary_mul(q3, cos_t, sin_t, mode)
        k3 = torch_npu.npu_rotary_mul(k3, cos_t, sin_t, mode)

        q = torch.concat([q1, q2, q3], dim=-1)
        k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


def apply_3drotary_pos_v5(q, k, freqs_cis, mat):
    '''
    本方案:自定义融合算子实现
    '''
    import rope_matrix
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
    cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)

    def rope(x, mat, cos, sin):
        x = rope_matrix.rope_matrix_kernel_bf16(x, mat, sin, cos)
        return x

    q = rope(q, mat, cos, sin)
    k = rope(k, mat, cos, sin)
    return q, k


class ROPE3D(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, q, k, freqs_cis, mat):
        return apply_3drotary_pos_v3(q, k, freqs_cis, mat)
    

def main():
    # init
    B, N, S, D = 1, 24, 28800, 128
    shape_lists_1d = [128]
    shape_lists_2d = [64, 64]
    shape_lists_3d = [44, 44, 40]

    q = torch.randn((B, N, S, D), dtype=torch.bfloat16, device='npu')
    k = torch.randn((B, N, S, D), dtype=torch.bfloat16, device='npu')
    freqs_cis_1d = []
    for shape in shape_lists_1d:
        sincos = torch.randn((S, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_1d.append([sincos, sincos])
    freqs_cis_2d = []
    for shape in shape_lists_2d:
        sincos = torch.randn((S, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_2d.append([sincos, sincos])
    freqs_cis_3d = []
    for shape in shape_lists_3d:
        sincos = torch.randn((S, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_3d.append([sincos, sincos])

    inter_mat_128 = get_interleave_matrix(128)
    inter_mat_64 = get_interleave_matrix(64)
    inter_mat_44 = get_interleave_matrix(44)
    inter_mat_40 = get_interleave_matrix(40)

    half_mat_128 = get_half_matrix(128)
    half_mat_64 = get_half_matrix(64)
    half_mat_44 = get_half_matrix(44)
    half_mat_40 = get_half_matrix(40)
    half_mat_44_44_40 = compose_3matrix(half_mat_44, half_mat_44, half_mat_40)

    mode = 'half' # or 'interleave or None

    #
    if g_ascend_env:
        version_path = "/usr/local/Ascend/ascend-toolkit/latest/version.cfg"
        os.system("sudo chmod -R 777 {}".format(version_path))
        with open(version_path, "r") as fn:
            env_info = fn.readlines()
        #
        env_info = [x for x in env_info if "runtime_running_version=" in x][0]
        env_info = env_info.strip().split("[")[1].split("]")[0]
        env_info = env_info.replace(":", "_")
        print("get env info: {}".format(env_info))
    else:
        env_info = "gpu"
    
    if g_ascend_env:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
        )
    else:
        experimental_config = None
    
    op = ROPE3D()
    import torchair
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    op = torch.compile(
        op,
        mode="default",
        backend=npu_backend,
        dynamic=False,
        fullgraph=False
    )

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiling_rope_matrix"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config) as prof:
        
        if mode == 'interleave':
            with torch.no_grad():
                for i in range(10):
                    # 3D rope
                    # if needed open these below
                    # outq1, outk1 = apply_3drotary_pos_v1(q, k, freqs_cis_3d, mode)
                    # outq2, outk2 = apply_3drotary_pos_v2(q, k, freqs_cis_3d, inter_mat_44, inter_mat_44, inter_mat_40)
                    # outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, inter_mat_128)
                    # outq3, outk3 = op(q, k, freqs_cis_3d, inter_mat_128)
                    # outq4, outk4 = apply_3drotary_pos_v4(q, k, freqs_cis_3d, mode)

                    XXX = torch.randn(3, 3).npu()
                    XXX = torch.pow(XXX, 2)
                    with record_function("3d rope v3"):
                        outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, inter_mat_128)
                        torch.cuda.synchronize()
        elif mode == 'half':
            with torch.no_grad():
                for i in range(10):
                    # 3D rope
                    # if needed open these below
                    # outq1, outk1 = apply_3drotary_pos_v1(q, k, freqs_cis_3d, mode)
                    # outq2, outk2 = apply_3drotary_pos_v2(q, k, freqs_cis_3d, half_mat_44, half_mat_44, half_mat_40)
                    # outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, half_mat_44_44_40)
                    # outq3, outk3 = op(q, k, freqs_cis_3d, half_mat_44_44_40)
                    # outq4, outk4 = apply_3drotary_pos_v4(q, k, freqs_cis_3d, mode)

                    XXX = torch.randn(3, 3).npu()
                    XXX = torch.pow(XXX, 2)
                    with record_function("3d rope v3 op"):
                        outq3, outk3 = op(q, k, freqs_cis_3d, half_mat_44_44_40)
                        torch.cuda.synchronize()
                    
                    XXX = torch.randn(3, 3).npu()
                    XXX = torch.pow(XXX, 2)
                    with record_function("3d rope v5"):
                        outq5, outk5 = apply_3drotary_pos_v5(q, k, freqs_cis_3d, half_mat_44_44_40)
                        torch.cuda.synchronize()

                    prof.step()

    # test precision
    with record_function("3d rope v5"):
        outq5, outk5 = apply_3drotary_pos_v5(q, k, freqs_cis_3d, half_mat_44_44_40)
        torch.cuda.synchronize()
    
    with record_function("3d rope v3"):
        outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, half_mat_44_44_40, debug=None, high_precision=True)
        torch.cuda.synchronize()
    
    # 精度验证
    print(f"q={q[0][0][0][:10]}")
    print(f"outq3={outq3[0][0][0][:10]}")
    print(f"outq5={outq5[0][0][0][:10]}")
    print((outq3 - outq5).max())
    print(outq3.max())
    print(outq5.max())

    denominator = torch.maximum(torch.abs(outq3), torch.abs(outq5))
    abs_error = torch.abs(torch.abs(outq3) - torch.abs(outq5))
    error = abs_error / (denominator + 1e-8)
    print(torch.minimum(abs_error, error).max())
    print((torch.minimum(abs_error, error) > 0).sum())
    
if __name__ == '__main__':
    main()