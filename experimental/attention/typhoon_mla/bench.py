
import torch 
import torch_npu
import numpy as np
import pytest 

from src.typhoon_mla import typhoon_mla_prepare, typhoon_mla_run

from tests.test_typhoonmla import TyphoonMLA
from tests.test_torchnpu_absorb import TorchNPUPagedMLA
from tests.utils import convert_absorb_to_naive, kv_to_paged, convert_to_dense

torch.set_printoptions(sci_mode=False)

if __name__=="__main__":
    n_heads = 128
    n_kv_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    softmax_scale = 1/np.sqrt(128)

    device = torch.device("npu:0")
    dtype = torch.bfloat16

    for bsz in [64, 128, 256, 512]:
        shared_seqlen = 4096
        nonshared_seqlen = 128

        block_size = 128
        seqlens = [shared_seqlen] + [nonshared_seqlen] * bsz
        
        test_typhoon_mla = TyphoonMLA(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)
        typhoonmla_elapsed = test_typhoon_mla.perf(warm_up=25, n_repeat=100)
        typhoonmla_tgr = bsz/(typhoonmla_elapsed*1e-3)*1e-3 #ktoken/s

        seqlens = [shared_seqlen + nonshared_seqlen] * bsz
        test_torchnpu_paged_mla = TorchNPUPagedMLA(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)
        torchnpu_elapsed = test_torchnpu_paged_mla.perf(warm_up=25, n_repeat=100)
        torchnpu_tgr = bsz/(torchnpu_elapsed*1e-3)*1e-3 #ktoken/s
        
        print(f"bsz: {bsz:<5} shared_kv_seqlen: {shared_seqlen:<5} nonshared_kv_seqlen: {nonshared_seqlen:<5} | TyphoonMLA (TBT): {typhoonmla_elapsed:.2f} ms   TorchNPU-Absorb (TBT): {torchnpu_elapsed:.2f} ms")