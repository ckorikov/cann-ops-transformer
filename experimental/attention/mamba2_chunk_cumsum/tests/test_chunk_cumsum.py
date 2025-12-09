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

from utils.utils import check_diff, profiling
# npu kernels
import npu_ops_transformer_ext


def mamba2_chunk_cumsum_forward(at, dt, dtbias, dtmask):
    B, C, L, H = dt.shape
    dt_add = dt.to(torch.float32) + torch.reshape(dtbias.to(torch.float32), (1, 1, 1, H))
    dtout = torch.log(1 + torch.exp(dt_add)) 
    dtout = torch.where(dt_add < 20, dtout, dt_add)
    dtout = torch.clamp(dtout, max=10000000.0) * dtmask.to(torch.float32)
    da = dtout * torch.reshape(at, (1, 1, 1, H))
    dacs = torch.cumsum(da, dim=2)
    dacs_chunk = torch.reshape(dacs[:, :, -1, :], (B, C, 1, H))
    return dtout, dacs, dacs_chunk



if __name__ == '__main__':
    B = 1
    C = 4
    H = 128
    G = 8
    L = 256
    
    device = torch.device("npu:0")

    tensor_at = torch.randn([H], dtype=torch.float32, device=device) * 0.2
    tensor_dt = torch.randn([B, C, L, H], dtype=torch.float16, device=device) * 0.2
    tensor_dtbias = torch.randn([H], dtype=torch.float16, device=device) * 0.2
    tensor_dtmask = torch.randn([B, C, L, H], dtype=torch.float16, device=device) * 0.2
    
    inputs = [tensor_at, tensor_dt, tensor_dtbias, tensor_dtmask]
    dtout, dacs, dacs_chunk = profiling(mamba2_chunk_cumsum_forward, inputs, 'TORCH')
    npu_dtout, npu_dacs, npu_dacs_chunk = profiling(torch.ops.npu_ops_transformer_ext.mambav2_chunk_cumsum, inputs, 'NPU_KERNEL')
    
    check_diff(dtout.cpu(), npu_dtout.cpu())
    check_diff(dacs.cpu(), npu_dacs.cpu())
    check_diff(dacs_chunk.cpu(), npu_dacs_chunk.cpu())
