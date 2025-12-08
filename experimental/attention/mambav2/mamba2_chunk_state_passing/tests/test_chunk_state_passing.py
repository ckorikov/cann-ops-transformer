import os
import numpy as np
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, suppress=True)
import torch
import torch_npu
import time
from rich.text import Text
from rich.style import Style
from rich.console import Console
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))

# npu kernel
import npu_ops_transformer_ext

def check_diff(x, y):
    diff = np.abs(x - y)
    arg = np.argmax(diff.flatten())
    print('Left:', x.flatten()[arg], 'Right:', y.flatten()[arg], 'Index:', arg, 'Pass: ', diff.max()<0.005)


def unpack_s4tos8(x):
    x1 = (((x & 0xf) << 4) >> 4)
    x2 = (x & np.uint8(0xf0).view(np.int8)) >> 4
    x_unpack = np.stack([x1, x2], axis=-1).flatten()
    return x_unpack

def chunk_state_passing_fwd(dacs, initial_states, states, cmtx, P=64, out_dtype=None):
    batch, nchunks, nheads, Z = states.shape
    batch, nchunks, chunk_size, ngroups, N = cmtx.shape
    assert Z == N*P
    
    cmtx = torch.repeat_interleave(cmtx.permute(0,1,3,2,4), nheads//ngroups, dim=2)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.zeros((batch, nchunks, nheads, Z), device=states.device, dtype=out_dtype)
    final_states = torch.zeros((batch, nheads, Z), device=states.device, dtype=torch.float32)

    tmp_states = torch.zeros((batch, nheads, Z), device=states.device, dtype=out_dtype)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, Z)
        tmp_states = initial_states

    out[:, 0, :, :] = tmp_states

    for c in range(nchunks):
        new_states = states[:, c, :, :]
        scale = torch.exp(dacs[:, c, :])
        tmp_states = scale[...,None] * tmp_states + new_states
        if c < nchunks - 1:
            out[:, c+1, :, :] = tmp_states
        else:
            final_state = tmp_states

    out = out.reshape(batch, nchunks, nheads, N, P).to(cmtx.dtype)
    final_state = final_state.reshape(batch, nheads, N, P)
    
    out_a = torch.matmul(cmtx.to(torch.float32), out.to(torch.float32))  # B, 4, 80, 256, 64  # original mm using fp16

    return out, final_state, out_a


if __name__ == '__main__':
    B = 1
    H = 128
    S = 1024
    C = 4
    L = 256
    G = 8
    N = 128
    P = 64
    Z = (N * P)
    
    device = torch.device("npu:6")

    dacs = np.random.random([((B * C) * H)]).astype(np.float32) * 0.5
    initmtx = np.random.random([((B * H) * Z)]).astype(np.float32) * 0.5
    statemtx = np.random.random([(((B * C) * H) * Z)]).astype(np.float32) * 0.5
    cmtx = np.random.random([(((B * S) * G) * N)]).astype(np.float16) * 0.5
    out_states = np.random.random([((((B * C) * H) * N) * P)]).astype(np.float32) * 0.5
    final_state = np.random.random([(((B * H) * N) * P)]).astype(np.float32) * 0.5
    out_bmm = np.random.random([((((B * C) * H) * L) * P)]).astype(np.float32) * 0.5

    ##### torch function golden ########
    tensor_dacs = torch.from_numpy(dacs).reshape(B,C,H).to(device)
    tensor_initmtx = torch.from_numpy(initmtx).reshape(B,H,Z).to(device)
    tensor_statemtx = torch.from_numpy(statemtx).reshape(B,C,H,Z).to(device)
    tensor_cmtx = torch.from_numpy(cmtx).reshape(B,C,L,G,N).to(device)

    out_states, final_state, out_bmm = chunk_state_passing_fwd(tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx)
    print(tensor_dacs.is_contiguous(), tensor_initmtx.is_contiguous(), tensor_statemtx.is_contiguous(), tensor_cmtx.is_contiguous())
    npu_out, npu_final_state = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state_passing(tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx)

    check_diff(final_state.cpu().numpy(), npu_final_state.cpu().numpy())
    check_diff(out_bmm.cpu().numpy(), npu_out.cpu().numpy())

    if 0:
        for i in range(5):
            out_states, final_state, out_bmm = chunk_state_passing_fwd(tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx)

        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            out_states, final_state, out_bmm = chunk_state_passing_fwd(tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx)
        torch.npu.synchronize()
        end = time.time()
        print("torch time: ", (end - start), 'ms')

        # npu kernel extension
        npu_out = torch.zeros_like(out_bmm)
        npu_final_state = torch.zeros_like(final_state)
        npu_workspace = torch.zeros([B*H*Z + 20*Z*3]).to(torch.float32).npu()
        
        print(f'{tensor_dacs.shape} {tensor_dacs.dtype}')
        print(f'{tensor_initmtx.shape} {tensor_initmtx.dtype}')
        print(f'{tensor_statemtx.shape} {tensor_statemtx.dtype}')
        print(f'{tensor_cmtx.shape} {tensor_cmtx.dtype}')
        print(f'{npu_out.shape} {npu_out.dtype}')
        print(f'{npu_final_state.shape} {npu_final_state.dtype}')
        print(f'{npu_workspace.shape} {npu_workspace.dtype}')
        
        
        for i in range(5):
            kernel_cust_mamba2_chunk_state_passing.kernel_cust_mamba2_chunk_state_passing(
                tensor_dacs,
                tensor_initmtx,
                tensor_statemtx,
                tensor_cmtx,
                npu_out,
                npu_final_state,
                npu_workspace
            )

        ######## PROFILING ########
        repeat = 10
        experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
        with torch_npu.profiler.profile(
            activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            experimental_config = experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./npu_results_B%d_S%d_H%d'%(B,C*L,H))
        ) as prof:

            kernel_cust_mamba2_chunk_state_passing.kernel_cust_mamba2_chunk_state_passing(
                tensor_dacs,
                tensor_initmtx,
                tensor_statemtx,
                tensor_cmtx,
                npu_out,
                npu_final_state,
                npu_workspace
            )

            torch.npu.synchronize()
            t1 = torch.npu.Event(enable_timing=True)
            t2 = torch.npu.Event(enable_timing=True)
            t1.record()
            for _ in range(repeat):
                kernel_cust_mamba2_chunk_state_passing.kernel_cust_mamba2_chunk_state_passing(
                tensor_dacs,
                tensor_initmtx,
                tensor_statemtx,
                tensor_cmtx,
                npu_out,
                npu_final_state,
                npu_workspace
            )
            t2.record()
            torch.npu.synchronize()
            elapsed = t1.elapsed_time(t2)

            print('NPU KERNEL kernel_cust_mamba2_chunk_state_passing TIME ELAPSED: %.1f'%(elapsed/repeat*1000), 'us')

            prof.step()
        ######## END OF PROFILING ########
            
        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            kernel_cust_mamba2_chunk_state_passing.kernel_cust_mamba2_chunk_state_passing(
                tensor_dacs,
                tensor_initmtx,
                tensor_statemtx,
                tensor_cmtx,
                npu_out,
                npu_final_state,
                npu_workspace
            )
        torch.npu.synchronize()
        end = time.time()
        print("kernel time: ", (end - start), 'ms')


        check_diff(final_state.cpu().numpy(), npu_final_state.cpu().numpy())
        check_diff(out_bmm.cpu().numpy(), npu_out.cpu().numpy())
    
