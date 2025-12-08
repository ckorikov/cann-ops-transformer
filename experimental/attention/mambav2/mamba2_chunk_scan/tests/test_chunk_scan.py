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
import math
import torch.nn.functional as F
from einops import rearrange, repeat
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))

import npu_ops_transformer_ext


def mamba2_chunkscan_custom_fwd(C, B, input_dacs, input_dttrans, input_mask, x, input_outa, input_dmtx, chunk_size=256, ngroups=8, headdim=64, z=None, seq_idx=None):
    #preprocess
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = C.shape
    assert C.shape == (batch, seqlen, ngroups, dstate)

    batch, seqlen, ngroups, dstate = B.shape
    nchunks = math.ceil(seqlen / chunk_size)
    padLen = nchunks * chunk_size - seqlen 
    B = F.pad(B, (0,0,0,0,0,padLen), "constant", 0)
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    # B = torch.repeat_interleave(B, nheads//ngroups, dim=3)
    B = torch.permute(B, (0, 1, 3, 4, 2))  

    batch, seqlen, nheads, headdim = x.shape
    x = F.pad(x, (0,0,0,0,0,padLen), "constant", 0)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    x = torch.permute(x, (0, 1, 2, 3, 4))

    C = F.pad(C, (0,0,0,0,0,padLen), "constant", 0)
    C = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    # C = torch.repeat_interleave(C, nheads//ngroups, dim=3)
    C = C.permute(0, 1, 3, 2, 4)  
    batch, nchunks, chunk_size, nheads, headdim = x.shape

    # c.dtype=half, b.dtype=half, accum.dtype=fp32, out.dtype=fp32
    cb = torch.matmul(C.to(torch.float32), B.to(torch.float32)).to(torch.float32) #bmtx和cmtx写反了 应该是cb，不是bc
    out_cb = cb
    # print(input_dacs.shape)
    cb = torch.repeat_interleave(cb, nheads//ngroups, dim=2)
    dummy = (input_dacs.reshape([batch, nchunks, nheads, chunk_size,1]) - input_dacs.reshape([batch, nchunks, nheads,1, chunk_size]))
    
    scale_b = torch.exp(dummy)
    val = cb * scale_b
    
    val = val * input_dttrans.reshape(batch,nchunks,nheads,1,chunk_size)
    val.masked_fill_(input_mask.bool(), 0.0)
    val = val.to(torch.float16)
    out_m = val.clone()
    
    out_b = torch.matmul(val.to(torch.float32), x.permute(0,1,3,2,4).to(torch.float32))
    
    out_a = input_outa * torch.exp(input_dacs.reshape([batch, nchunks, nheads, chunk_size, 1]))
    
    out = out_a + out_b

    x_res = x.to(torch.float32) * input_dmtx[..., None].to(torch.float32)
    out += x_res.permute(0,1,3,2,4)
    out = out.transpose(2,3)
    y = out.reshape(batch, -1, nheads, headdim)

    return out_cb, out_m, out_b, y


if __name__ == '__main__':
    B = 1
    S = 1024
    C = 4
    H = 128
    L = 256
    G = 8
    N = 128
    P = 64

    device = torch.device("npu:6")
    
    cmtx = np.random.random([(B * C * L * G * N)]).astype(np.float16) * 0.5
    bmtx = np.random.random([(B * C * L * G * N)]).astype(np.float16) * 0.5
    xmtx = np.random.random([(B * C * L * H * P)]).astype(np.float16) * 0.5
    dmtx = np.random.random([(H)]).astype(np.float16) * 0.5
    input_dacs = np.random.random([(B * C * H * L)]).astype(np.float32) * 0.5
    input_dtout = np.random.random([(B * C * H * L)]).astype(np.float32) * 0.5
    input_outa = np.random.random([(B * C * H * L * P)]).astype(np.float32) * 0.5
    dmtx = np.random.random([H]).astype(np.float16) * 0.5
    
    
    tensor_cmtx = torch.from_numpy(cmtx).reshape(B,C*L,G*N).to(device)
    tensor_bmtx = torch.from_numpy(bmtx).reshape(B,C*L,G*N).to(device)
    tensor_xmtx = torch.from_numpy(xmtx).reshape(B,C*L,H*P).to(device)
    tensor_dacs = torch.from_numpy(input_dacs).reshape(B,C,H,L).to(device)
    tensor_dtout = torch.from_numpy(input_dtout).reshape(B,C,H,L).to(device)
    tensor_outa = torch.from_numpy(input_outa).reshape(B,C,H,L,P).to(device)
    tensor_dmtx = torch.from_numpy(dmtx).to(device)
    mask = torch.triu(torch.ones(L, L), diagonal=1).to(device)
    
    out_cb, out_m, out_b, out_y = mamba2_chunkscan_custom_fwd(tensor_cmtx, tensor_bmtx, tensor_dacs, tensor_dtout, mask, tensor_xmtx, tensor_outa, tensor_dmtx, chunk_size=L)

    npu_y = torch.ops.npu_ops_transformer_ext.mambav2_chunk_scan(
                tensor_cmtx.reshape(B,C,L,G,N),
                tensor_bmtx,
                tensor_xmtx,
                tensor_dmtx,
                tensor_outa,
                tensor_dacs,
                tensor_dtout  
            )
    check_diff(out_y.cpu().numpy(), npu_y.cpu().numpy())


    if 0:
        for i in range(10):
            out_cb, out_m, out_b, out_y = mamba2_chunkscan_custom_fwd(tensor_cmtx, tensor_bmtx, tensor_dacs, tensor_dtout, mask, tensor_xmtx, tensor_outa, tensor_dmtx, chunk_size=L)
            
        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            out_cb, out_m, out_b, out_y = mamba2_chunkscan_custom_fwd(tensor_cmtx, tensor_bmtx, tensor_dacs, tensor_dtout, mask, tensor_xmtx, tensor_outa, tensor_dmtx, chunk_size=L)
        torch.npu.synchronize()
        end = time.time()
        print(f'torch time: {(end - start)} ms')
        
        
        npu_cb = torch.zeros_like(out_cb)
        npu_m = torch.zeros_like(out_m)
        npu_b = torch.zeros_like(out_b)
        npu_y = torch.zeros_like(out_y)
        npu_workspace = torch.empty([20 * L * L]).to(torch.float32).npu()
        
        print(f'{tensor_cmtx.reshape(B,C,L,G,N).shape}, {tensor_cmtx.dtype}')
        print(f'{tensor_bmtx.shape}, {tensor_bmtx.dtype}')
        print(f'{tensor_xmtx.shape}, {tensor_xmtx.dtype}')
        print(f'{tensor_dmtx.shape}, {tensor_dmtx.dtype}')
        print(f'{tensor_outa.shape}, {tensor_outa.dtype}')
        print(f'{tensor_dacs.shape}, {tensor_dacs.dtype}')
        print(f'{tensor_dtout.shape}, {tensor_dtout.dtype}')
        print(f'{npu_cb.shape}, {npu_cb.dtype}')
        print(f'{npu_m.shape}, {npu_m.dtype}')
        print(f'{npu_b.shape}, {npu_b.dtype}')
        print(f'{npu_y.shape}, {npu_y.dtype}')
        print(f'{npu_workspace.shape} {npu_workspace.dtype}')
        
        for i in range(5):
            kernel_cust_mamba2_chunk_scan.kernel_cust_mamba2_chunk_scan(
                tensor_cmtx.reshape(B,C,L,G,N),
                tensor_bmtx,
                tensor_xmtx,
                tensor_dmtx,
                tensor_outa,
                tensor_dacs,
                tensor_dtout,
                npu_cb,
                npu_m,
                npu_b,
                npu_y,
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

            kernel_cust_mamba2_chunk_scan.kernel_cust_mamba2_chunk_scan(
                tensor_cmtx.reshape(B,C,L,G,N),
                tensor_bmtx,
                tensor_xmtx,
                tensor_dmtx,
                tensor_outa,
                tensor_dacs,
                tensor_dtout,
                npu_cb,
                npu_m,
                npu_b,
                npu_y,
                npu_workspace
            )

            torch.npu.synchronize()
            t1 = torch.npu.Event(enable_timing=True)
            t2 = torch.npu.Event(enable_timing=True)
            t1.record()
            for _ in range(repeat):
                kernel_cust_mamba2_chunk_scan.kernel_cust_mamba2_chunk_scan(
                    tensor_cmtx.reshape(B,C,L,G,N),
                    tensor_bmtx,
                    tensor_xmtx,
                    tensor_dmtx,
                    tensor_outa,
                    tensor_dacs,
                    tensor_dtout,
                    npu_cb,
                    npu_m,
                    npu_b,
                    npu_y,
                    npu_workspace
                )
            t2.record()
            torch.npu.synchronize()
            elapsed = t1.elapsed_time(t2)

            print('NPU KERNEL kernel_cust_mamba2_chunk_scan TIME ELAPSED: %.1f'%(elapsed/repeat*1000), 'us')

            prof.step()
        ######## END OF PROFILING ########
        torch.npu.synchronize()
        start = time.time()
        for i in range(1000):
            kernel_cust_mamba2_chunk_scan.kernel_cust_mamba2_chunk_scan(
                tensor_cmtx.reshape(B,C,L,G,N),
                tensor_bmtx,
                tensor_xmtx,
                tensor_dmtx,
                tensor_outa,
                tensor_dacs,
                tensor_dtout,
                npu_cb,
                npu_m,
                npu_b,
                npu_y,
                npu_workspace
            )
        torch.npu.synchronize()
        end = time.time()
        print(f'kernel time: {(end - start)} ms')
        
        # check_diff(gt_cbmtx, npu_cb.cpu().numpy().reshape(-1), 'cube0')
        # check_diff(gt_mmtx, npu_m.cpu().numpy().reshape(-1), 'vec0')
        # check_diff(gt_outmtx, npu_b.cpu().numpy().reshape(-1), 'cube1')
        # check_diff(gt_out_y, npu_y.cpu().numpy().reshape(-1), 'vec1')
    
        check_diff(out_cb.cpu().numpy(), npu_cb.cpu().numpy(), 'cube0')
        check_diff(out_m.cpu().numpy(), npu_m.cpu().numpy(), 'vec0')
        check_diff(out_b.cpu().numpy(), npu_b.cpu().numpy(), 'cube1')
        check_diff(out_y.cpu().numpy(), npu_y.cpu().numpy(), 'vec1')