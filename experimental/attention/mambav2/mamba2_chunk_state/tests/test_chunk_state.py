import os
import numpy as np
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, suppress=True)
from rich.text import Text
from rich.style import Style
from rich.console import Console
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))
import torch, torch_npu
import time

# npu kernel
import npu_ops_transformer_ext

def check_diff(x, y):
    diff = np.abs(x - y)
    arg = np.argmax(diff.flatten())
    print('Left:', x.flatten()[arg], 'Right:', y.flatten()[arg], 'Index:', arg)

def unpack_s4tos8(x):
    x1 = (((x & 0xf) << 4) >> 4)
    x2 = (x & np.uint8(0xf0).view(np.int8)) >> 4
    x_unpack = np.stack([x1, x2], axis=-1).flatten()
    return x_unpack

def mamba2_chunk_state_forward(dtout, dacs, bt, xt, num_repeats):

    B, C, L, H = dacs.shape
    
    da_sub = torch.reshape(dacs[:,:,-1,:], (B, C, 1, H)) - dacs
    da = torch.exp(da_sub) * dtout  # BCLH
    
    bt_repeated_tensor = torch.repeat_interleave(bt, num_repeats, dim=3).float()
    dab = bt_repeated_tensor * torch.reshape(da, (B, C, L, H, 1))   
    out = dab.permute(0,1,3,4,2) @ xt.to(torch.float32).permute(0, 1, 3, 2, 4)
    
    return out
    

if __name__ == '__main__':
    B = 1
    C = 4
    H = 128
    G = 8
    L = 256
    N = 128
    P = 64
    BASEH=8
    CBASEM=64
    
    device = torch.device('npu:6')
    # op_path = '../../mamba2_nemotronh_ops/ascautogenv2-mamba2_chunk_state'
    # dtout_mtx = np.fromfile(os.path.join(op_path, "./workspace/input/input_dtout_mtx.bin"), dtype=np.float32)
    # dacs_mtx = np.fromfile(os.path.join(op_path, "./workspace/input/input_dacs_mtx.bin"), dtype=np.float32)
    # bt_mtx = np.fromfile(os.path.join(op_path, "./workspace/input/input_bt_mtx.bin"), dtype=np.float16)
    # xt_mtx = np.fromfile(os.path.join(op_path, "./workspace/input/input_xt_mtx.bin"), dtype=np.float16)
    # out_mtx = np.fromfile(os.path.join(op_path, "./workspace/output/output_out_mtx.bin"), dtype=np.float32)
    # da_out_mtx = np.fromfile(os.path.join(op_path, "./workspace/output/output_da_out_mtx.bin"), dtype=np.float16)
    # workspace = np.fromfile(os.path.join(op_path, "./workspace/output/output_workspace.bin"), dtype=np.float32)
    
    dtout_mtx = np.random.random([(((B * C) * L) * H)]).astype(np.float32) * 0.5
    dacs_mtx = np.random.random([(((B * C) * L) * H)]).astype(np.float32) * 0.5
    bt_mtx = np.random.random([((((B * C) * L) * G) * N)]).astype(np.float16) * 0.5
    xt_mtx = np.random.random([((((B * C) * L) * H) * P)]).astype(np.float16) * 0.5
    out_mtx = np.random.random([((((B * C) * H) * N) * P)]).astype(np.float32) * 0.5 
       
    print('torch compute')
    dtout_mtx = dtout_mtx.reshape([B, C, L, H])
    dacs_mtx = dacs_mtx.reshape([B, C, L, H])
    bt_mtx = bt_mtx.reshape([B, C, L, G, N])
    xt_mtx = xt_mtx.reshape([B, C, L, H, P])
    
    tensor_dtout = torch.from_numpy(dtout_mtx).to(device)
    tensor_dacs = torch.from_numpy(dacs_mtx).to(device)
    tensor_bt = torch.from_numpy(bt_mtx).to(device)
    tensor_xt = torch.from_numpy(xt_mtx).to(device)
    assert H%G==0

    outmtx = mamba2_chunk_state_forward(tensor_dtout, tensor_dacs, tensor_bt, tensor_xt, H//G)
    print(tensor_dtout.device, tensor_dacs.device, tensor_bt.device, tensor_xt.device)
    print(tensor_dtout.is_contiguous(), tensor_dacs.is_contiguous(), tensor_bt.is_contiguous(), tensor_xt.is_contiguous())
    npu_out = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state(tensor_dtout, tensor_dacs, tensor_bt, tensor_xt)

    check_diff(outmtx.cpu().numpy(), npu_out.cpu().numpy())

    if 0:
        for i in range(5):
            outmtx = mamba2_chunk_state_forward(tensor_dtout, tensor_dacs, tensor_bt, tensor_xt, H//G)

        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            outmtx = mamba2_chunk_state_forward(tensor_dtout, tensor_dacs, tensor_bt, tensor_xt, H//G)
        torch.npu.synchronize()
        end = time.time()

        print('torch time: ', (end - start), 'ms')
        
        npu_out = torch.zeros_like(outmtx)
        npu_workspace = torch.zeros_like(tensor_workspace)
        # npu_workspace = torch.empty([20*L*BASEH + 20*BASEH*L*CBASEM*3]).to(torch.float32)
        
        print(f'{tensor_dtout.shape}  {tensor_dtout.dtype}')
        print(f'{tensor_dacs.shape}   {tensor_dacs.dtype}')
        print(f'{tensor_bt.shape}     {tensor_bt.dtype}')
        print(f'{tensor_xt.shape}     {tensor_xt.dtype}')
        print(f'{npu_out.shape}       {npu_out.dtype}')
        print(f'{npu_workspace.shape} {npu_workspace.dtype}')
        
        for i in range(5):
            kernel_cust_mamba2_chunk_state.kernel_cust_mamba2_chunk_state(
                tensor_dtout,
                tensor_dacs, 
                tensor_bt,
                tensor_xt,
                npu_out,
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

            kernel_cust_mamba2_chunk_state.kernel_cust_mamba2_chunk_state(
                tensor_dtout,
                tensor_dacs, 
                tensor_bt,
                tensor_xt,
                npu_out,
                npu_workspace
            )
            
            torch.npu.synchronize()
            t1 = torch.npu.Event(enable_timing=True)
            t2 = torch.npu.Event(enable_timing=True)
            t1.record()
            for _ in range(repeat):
                kernel_cust_mamba2_chunk_state.kernel_cust_mamba2_chunk_state(
                    tensor_dtout,
                    tensor_dacs, 
                    tensor_bt,
                    tensor_xt,
                    npu_out,
                    npu_workspace
                )
            t2.record()
            torch.npu.synchronize()
            elapsed = t1.elapsed_time(t2)

            print('NPU KERNEL kernel_cust_mamba2_chunk_state TIME ELAPSED: %.1f'%(elapsed/repeat*1000), 'us')

            prof.step()
        ######## END OF PROFILING ########
        
        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            kernel_cust_mamba2_chunk_state.kernel_cust_mamba2_chunk_state(
                tensor_dtout,
                tensor_dacs, 
                tensor_bt,
                tensor_xt,
                npu_out,
                npu_workspace
            )
        torch.npu.synchronize()
        end = time.time()

        print('kernel time: ', (end - start), 'ms')
        
        check_diff(outmtx.cpu().numpy(), npu_out.cpu().numpy())
