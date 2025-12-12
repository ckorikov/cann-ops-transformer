
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
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    softmax_scale = 1/np.sqrt(128)
    block_size = 128

    device = torch.device("npu:0")
    dtype = torch.bfloat16

    for n_heads in [64, 128]:
        for bsz in [128, 256, 512]:
            for shared_seqlen in [4*1024, 8*1024, 16*1024]:
                for nonshared_seqlen in [256, 1024, 4096]:
                    seqlens = [shared_seqlen] + [nonshared_seqlen] * bsz
                    
                    test_typhoon_mla = TyphoonMLA(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)
                    typhoonmla_elapsed = test_typhoon_mla.perf(warm_up=25, n_repeat=100)
                    typhoonmla_tgr = bsz/(typhoonmla_elapsed*1e-3)*1e-3 #ktoken/s

                    seqlens = [shared_seqlen + nonshared_seqlen] * bsz
                    test_torchnpu_paged_mla = TorchNPUPagedMLA(bsz, seqlens, n_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, softmax_scale, device, dtype)
                    torchnpu_elapsed = test_torchnpu_paged_mla.perf(warm_up=25, n_repeat=100)
                    torchnpu_tgr = bsz/(torchnpu_elapsed*1e-3)*1e-3 #ktoken/s

                    speedup = typhoonmla_tgr / torchnpu_tgr
                    print(f"batch: {bsz:<5}  shared_seqlen: {shared_seqlen:<5}  nonshared_seqlen: {nonshared_seqlen:<5}  headnum: {n_heads:<5}  |  TyphoonMLA: {typhoonmla_tgr:>6.2f} ktoken/s  TorchNPU-Absorb: {torchnpu_tgr:>6.2f} ktoken/s  Speedup: {speedup:.2f}x")