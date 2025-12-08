import numpy as np
import os
import torch
import torch.nn.functional as F
import torch_npu
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, suppress=True)
from rich.text import Text
from rich.style import Style
from rich.console import Console
import time
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))

## npu kernels
import npu_ops_transformer_ext
# if libc10.so cannot be found:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

def check_diff(x, y):
    if isinstance(x, torch.Tensor):
        diff = torch.abs(x - y)#.cpu().detach().numpy()
        arg = np.argmax(diff.flatten())
        print('Left:', x.cpu().detach().numpy().flatten()[arg], 'Right:', y.cpu().detach().numpy().flatten()[arg], 'Index:', arg, 'diff max: ', diff.max(), 'Pass: ', diff.max()<0.001)
    elif isinstance(x, np.ndarray):
        diff = np.abs(x - y)
        arg = np.argmax(diff.flatten())
        print('Left:', x.flatten()[arg], 'Right:', y.flatten()[arg], 'Index:', arg, 'diff max: ', diff.max(), 'Pass: ', diff.max()<0.001)

# def rms_norm_gated_ref_clean(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
#     dtype = x.dtype
#     N = x.shape[-1]
#     weight = weight.float()
#     x = x.float()
#     z = z.float() 

#     B, S, D = x.shape
#     x_group = x.reshape(B, S, -1, group_size)
#     rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
#     out = (x_group * rstd).reshape(B,S,D) * weight

#     out *= F.silu(z)

#     return out.to(dtype)

def rms_norm_gated_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=False, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        #x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        B, S, D = x.shape
        x_group = x.reshape(B, S, -1, group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        #out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        out = (x_group * rstd).reshape(B,S,D) * weight
        if bias is not None:
            out = out + bias
    # if z is not None and norm_before_gate:
    #     out *= F.silu(z)
    return out.to(dtype)

if __name__ == '__main__':
    B = 1
    S = 1024
    D = 8192 #16384
    G = 8
    E = 1e-05
    
    # kernel_path = '../../mamba2_nemotronh_ops/ascautogen-mamba2_rmsnormgated'
    # xmtx = np.fromfile(os.path.join(kernel_path, "./workspace/input/input_xmtx.bin"), dtype=np.float16)
    # zmtx = np.fromfile(os.path.join(kernel_path, "./workspace/input/input_zmtx.bin"), dtype=np.float16)
    # wmtx = np.fromfile(os.path.join(kernel_path, "./workspace/input/input_wmtx.bin"), dtype=np.float16)
    # outmtx = np.fromfile(os.path.join(kernel_path, "./workspace/output/output_outmtx.bin"), dtype=np.float16)
    # workspace = np.fromfile(os.path.join(kernel_path, './workspace/output/output_workspace.bin'), dtype=np.float16)
    
    xmtx = np.random.random([((B * S) * D)]).astype(np.float32) * 0.5
    zmtx = np.random.random([((B * S) * D)]).astype(np.float32) * 0.5
    wmtx = np.random.random([D]).astype(np.float32) * 0.5
    outmtx = np.random.random([((B * S) * D)]).astype(np.float32) * 0.5
    
    workspace = np.random.random([4*1024*1024]).astype(np.float32) * 0.5
    
    #### debug llm ####
#     if 1:
#         x = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_x.pt').to(torch.float32).cpu().detach().numpy()
#         w = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_w.pt').to(torch.float32).cpu().detach().numpy()
#         z = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_z.pt').to(torch.float32).cpu().detach().numpy()

#         xmtx = torch.from_numpy(x).reshape(B, S, D).npu()
#         wmtx = torch.from_numpy(w).npu()
#         zmtx = torch.from_numpy(z).reshape(B, S, D).npu()
#     else:
#         xmtx = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_x.pt')
#         wmtx = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_w.pt')
#         zmtx = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_z.pt')
        
#     print(xmtx.is_contiguous(), wmtx.is_contiguous(), zmtx.is_contiguous())
#     xmtx = xmtx.contiguous()
#     zmtx = zmtx.contiguous()
    
#     llm_torch = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_gt_torch.pt').cpu().detach()
#     llm_npu = torch.load('/home/ma-user/work/h00515348/nemotron-h/llm_quant_sg/debug_norm_npu_out.pt').cpu().detach()
    
#     # print(xmtx.shape, xmtx.dtype, xmtx.device)
#     # print(wmtx.shape, wmtx.dtype, wmtx.device)
#     # print(zmtx.shape, zmtx.dtype, zmtx.device)
#     # print(llm_torch.shape, llm_torch.dtype, llm_torch.device)
#     # print(llm_npu.shape, llm_npu.dtype, llm_npu.device)
    
#     new_torch = rms_norm_gated_ref(xmtx, wmtx, None, zmtx, eps=E, group_size=D//G)
#     kernel_out = torch.zeros_like(llm_npu).npu()
#     tensor_workspace = torch.from_numpy(workspace).npu()
#     kernel_cust_rmsnormgated.kernel_cust_rmsnormgated(xmtx, zmtx.to(torch.float32), wmtx.to(torch.float32), kernel_out, tensor_workspace, G, E)
    
#     # print(new_torch.shape, new_torch.device, llm_torch.shape, llm_torch.device)
#     # print(kernel_out.shape, kernel_out.device, llm_npu.shape, llm_npu.device)
#     check_diff(new_torch.cpu().detach(), llm_torch.cpu().detach())
#     check_diff(kernel_out.cpu().detach(), llm_npu.cpu().detach())
#     check_diff(new_torch.cpu().detach(), kernel_out.cpu().detach())

    
    tensor_x = torch.from_numpy(xmtx).reshape(B, S, D).npu()
    tensor_w = torch.from_numpy(wmtx).npu()
    tensor_z = torch.from_numpy(zmtx).reshape(B, S, D).npu()
    tensor_workspace = torch.from_numpy(workspace).npu()
    assert D%G==0

    outmtx = rms_norm_gated_ref(tensor_x, tensor_w, None, tensor_z, eps=E, group_size=D//G)
    kernel_out = torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(tensor_x, tensor_z, tensor_w, G, E)

    check_diff(outmtx.cpu().numpy(), kernel_out.cpu().numpy())

    if 0:
        
        for i in range(5):
            outmtx = rms_norm_gated_ref(tensor_x, tensor_w, None, tensor_z, eps=E, group_size=D//G)
        
        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            outmtx = rms_norm_gated_ref(tensor_x, tensor_w, None, tensor_z, eps=E, group_size=D//G)
        torch.npu.synchronize()
        end = time.time()
        
        print('torch time: ', (end - start), 'ms')
        
        print(tensor_x.shape, tensor_x.dtype)
        print(tensor_z.shape, tensor_z.dtype)
        print(tensor_w.shape, tensor_w.dtype)
        print(outmtx.shape, outmtx.dtype)
        print(tensor_workspace.shape, tensor_workspace.dtype)
        
        kernel_out = torch.zeros_like(outmtx).npu()
        for i in range(5):
            kernel_out = torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(tensor_x, tensor_z, tensor_w, G, E)
        
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
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./npu_results_B%d_S%d_D%d'%(B,S,D))
        ) as prof:

            kernel_out = torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(tensor_x, tensor_z, tensor_w, G, E)
            
            torch.npu.synchronize()
            t1 = torch.npu.Event(enable_timing=True)
            t2 = torch.npu.Event(enable_timing=True)
            t1.record()
            for _ in range(repeat):
                kernel_out = torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(tensor_x, tensor_z, tensor_w, G, E)
            t2.record()
            torch.npu.synchronize()
            elapsed = t1.elapsed_time(t2)

            print('NPU KERNEL kernel_cust_rmsnormgated TIME ELAPSED: %.1f'%(elapsed/repeat*1000), 'us')

            prof.step()
        ######## END OF PROFILING ########
        
        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            kernel_out = torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(tensor_x, tensor_z, tensor_w, G, E)
        torch.npu.synchronize()
        end = time.time()
        
        print('kernel time: ', (end - start), 'ms')
        
        
        # check_diff(out_gt, outmtx.cpu().numpy())
        check_diff(outmtx.cpu().numpy(), kernel_out.cpu().numpy())
        
        
#         # running profile
#         experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
#         with torch_npu.profiler.profile(
#             activities=[
#                     torch_npu.profiler.ProfilerActivity.CPU,
#                     torch_npu.profiler.ProfilerActivity.NPU
#                 ],
#             schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
#             experimental_config = experimental_config,
#             on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./npu_results_rmsnormgated')
#         ) as prof:

#             # warm up
#             kernel_cust_rmsnormgated.kernel_cust_rmsnormgated(tensor_x, tensor_z, tensor_w, kernel_out, tensor_workspace, G, E)
#             torch.npu.synchronize()
#             t1 = torch.npu.Event(enable_timing=True)
#             t2 = torch.npu.Event(enable_timing=True)
#             t1.record()
#             for _ in range(10):
#                 kernel_cust_rmsnormgated.kernel_cust_rmsnormgated(tensor_x, tensor_z, tensor_w, kernel_out, tensor_workspace, G, E)
#             t2.record()
#             torch.npu.synchronize()
#             elapsed = t1.elapsed_time(t2)

#             print('(RMSNORMGATED CUST) TIME ELAPSED: %.1f'%(elapsed/10*1000), 'us')

#             prof.step()