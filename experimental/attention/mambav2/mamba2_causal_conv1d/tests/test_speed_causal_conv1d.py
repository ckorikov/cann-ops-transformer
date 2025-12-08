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
import torch.nn.functional as F
import time

from utils.utils import check_diff, profiling
# npu kernel
import npu_ops_transformer_ext

def causal_conv1d_fwd(x, weight, bias, B, D, S, W):
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)

    x = F.conv1d(
        x,
        weight=weight,
        bias=bias,
        stride=1,
        padding=W-1,
        dilation=1,
        groups=D,
    )
    
    x = x[..., :S] 
    x = F.silu(x)
    return x
    
def mamba_causal_conv1d_npu(
    x,
    weight,
    bias
):
    npu_out = torch.ops.npu_ops_transformer_ext.mambav2_causal_conv1d(
                    x,
                    weight, 
                    bias,
                )
    
    return npu_out

if __name__ == '__main__':
    B = 1
    D = 10240
    S = 1024
    W = 4

    device = torch.device("npu:0")
       
    print('torch compute')
    tensor_xmtx = torch.randn([B, D, S], dtype=torch.float16, device=device) * 0.5
    tensor_wmtx = torch.randn([D, 1, W], dtype=torch.float16, device=device) * 0.5
    tensor_bias = torch.randn([D], dtype=torch.float16, device=device) * 0.5

    outmtx = profiling(causal_conv1d_fwd,
                       [tensor_xmtx, tensor_wmtx, tensor_bias, B, D, S, W],
                       'TORCH')
    npu_out = profiling(mamba_causal_conv1d_npu,
                        [tensor_xmtx, tensor_wmtx, tensor_bias],
                        'NPU_KERNEL')
    check_diff(outmtx, npu_out)



