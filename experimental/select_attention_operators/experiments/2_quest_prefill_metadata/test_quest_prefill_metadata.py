"""
Single input testing (for debugging) -> Run this file with python <filename>
Wide range testing (for validation) -> Run this file with pytest <filename>
"""
import pytest
import torch
import torch_npu
from itertools import product
from select_attn_prefill_ops import quest_prefill_metadata
from ref_quest_prefill_metadata import ref_quest_prefill_metadata
from gen_data_quest_prefill_metadata import gen_quest_prefill_inputs, compare_tensors, ceil_div


device = "npu:0"
BLOCK_SIZE_DEFAULT = 128
D_DEFAULT = 128


# --------------------------------------------------------------------------- #
# Central test worker  (assertion crashes <--> test failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="internal worker - not called directly by pytest")
def test_prefill_kernel(DTYPE:torch.dtype, B: int, N: int, BLOCK_SIZE: int, D: int,
                        MKBPR: int = 128,
                        MMBPR: int = 1,
                        SSAR: bool = True,
                        verbose: bool = False) -> None:
    """Run reference vs Ascend-C and assert bit-accurate match.
    
    Arguments:
        B - batch size
        N - number of KV heads
        BLOCK_SIZE - nmuumbe of tokens per block (equal for metadata block and 
                     kv-cahce block)
        D - head dimeansion
        MKBPR - maximum number of KV-cache blocks per request in a batch    
        MMBPR - maximum number of metadata blocks per request in a batch    
        SSAR - "same_seq_len_all_request" 
               if true, the the data generated will have same sequence length 
               for all request in the batch.
               if false: each request wll have a randomly number of tokens.
    
    """

    MMBPR = ceil_div(MKBPR, BLOCK_SIZE)
    num_kv_blocks = B * MKBPR  # total number of KV-cache blocks for the entire job (all requests together)
    num_meta_blocks = B * MMBPR  # total number of metadata blocks for the entire job (all requests together)

    # ---- generate input/output tensors ---- #
    k_cache, block_tables, seq_lens, meta_ids, max_out, min_out = gen_quest_prefill_inputs(
        B, N, BLOCK_SIZE, D,
        num_kv_blocks=num_kv_blocks,
        num_meta_blocks=num_meta_blocks,
        MKBPR=MKBPR,
        MMBPR=MMBPR,
        same_seq_len_all_reqs=SSAR, 
        dtype=DTYPE,
        device=device)
    
    # ---- reference kernel ---- #
    max_ref = min_out.clone() 
    min_ref = max_out.clone()
    ref_quest_prefill_metadata(
        k_cache, block_tables, seq_lens, meta_ids,
        max_ref, min_ref)

    # ---- custom kernel ---- #
    quest_prefill_metadata(
        k_cache, block_tables, seq_lens, meta_ids,
        max_out, min_out)

    # ---- compare ---- #
    max_ok = compare_tensors(max_ref, max_out)
    min_ok = compare_tensors(min_ref, min_out)
    assert max_ok and min_ok, "maxblocks or minblocks mismatch"
    if verbose:
        print(" ==================== maxblocks =================== ")
        print(f"{max_ref=}")
        print(f"{max_out=}")
        print(" ==================== minblocks =================== ")
        print(f"{min_ref=}")
        print(f"{min_out=}")    
        print(" ==================== SUMMARY =================== ")
        print(f"{B=} {N=} {BLOCK_SIZE=} {D=} {MKBPR=} {MMBPR=} {DTYPE=} {num_kv_blocks=} {num_meta_blocks=}")
        print("maxblocks - ", end='')
        compare_tensors(max_ref, max_out)    
        print("minblocks - ", end='')
        compare_tensors(min_ref, min_out)    


# --------------------------------------------------------------------------- #
# 3.  Param-set builder  (same pattern as block-select file)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="internal helper")
def construct_prefill_parameter_sets(B_VALUES, N_VALUES, MKBPR_VALUES, SSAR_VALUES):
    """
    Build legal (B,N,MKBPR) tuples.
    BLOCK_SIZE & D are global constants.
    """
    sets = []
    for b in B_VALUES:
        for n in N_VALUES:
            for MKBPR in MKBPR_VALUES:
                sets.append((b, n, MKBPR))
    return sets


# --------------------------------------------------------------------------- #
# Test 1 – Basic functionality
# ---------------------------------------------------------------------------#
DTYPE_BASIC = [torch.float16, torch.bfloat16]
B_BASIC = [1, 2]
N_BASIC = [4, 8]
MKBPR_BASIC = [1, 64, 126, 128]
SSAR_BASIC = [True, False]

@pytest.mark.parametrize(
    "DTYPE, B, N, MKBPR, SSAR", product(DTYPE_BASIC, B_BASIC, N_BASIC, MKBPR_BASIC, SSAR_BASIC),
    ids=[f"{DTYPE=},{B=},{N=},{MKBPR=},{SSAR=}" for DTYPE, B, N, MKBPR, SSAR in product(DTYPE_BASIC, B_BASIC, N_BASIC, MKBPR_BASIC, SSAR_BASIC)]
)
@torch.inference_mode()
def test_basic_functionality(DTYPE: torch.dtype, B: int, N: int, MKBPR: int, SSAR: int):
    test_prefill_kernel(DTYPE, B, N, BLOCK_SIZE_DEFAULT, D_DEFAULT, MKBPR, SSAR)


# --------------------------------------------------------------------------- #
# Test 2 – Edge cases
# ---------------------------------------------------------------------------#
DTYPE_EDGE = [torch.float16, torch.bfloat16]
B_EDGE = [1, 2]
N_EDGE = [1, 2, 4, 7, 8, 9, 16, 21, 32, 33]
MKBPR_EDGE = [1, 2, 3, 63, 64, 65, 126, 127, 128, 129, 150, 255, 256, 257]
SSAR_EDGE = [True, False]

@pytest.mark.parametrize(
    "DTYPE, B, N, MKBPR, SSAR", product(DTYPE_EDGE, B_EDGE, N_EDGE, MKBPR_EDGE, SSAR_EDGE),
    ids=[f"{DTYPE=},{B=},{N=},{MKBPR=},{SSAR=}" for DTYPE, B, N, MKBPR, SSAR in product(DTYPE_EDGE, B_EDGE, N_EDGE, MKBPR_EDGE, SSAR_EDGE)]
)
@torch.inference_mode()
def test_edge_cases(DTYPE: torch.dtype, B: int, N: int, MKBPR: int, SSAR: int):
    test_prefill_kernel(DTYPE, B, N, BLOCK_SIZE_DEFAULT, D_DEFAULT, MKBPR, SSAR)


# --------------------------------------------------------------------------- #
# Test 3 – Test Large sequence
# ---------------------------------------------------------------------------#
DTYPE_LS = [torch.float16, torch.bfloat16]
B_LS = [1, 2, 4, 8]
N_LS = [2, 4, 8]
MKBPR_LS = [1, 64, 126, 128, 130, 135, 150, 151, 170, 200, 210, 211, 212, 256, 300, 400, 512]
SSAR_LS = [True, False]

@pytest.mark.parametrize(
    "DTYPE, B, N, MKBPR, SSAR", product(DTYPE_LS, B_LS, N_LS, MKBPR_LS, SSAR_LS),
    ids=[f"{DTYPE=},{B=},{N=},{MKBPR=},{SSAR=}" for DTYPE, B, N, MKBPR, SSAR in product(DTYPE_LS, B_LS, N_LS, MKBPR_LS, SSAR_LS)]
)
@torch.inference_mode()
def test_large_lequence(DTYPE: torch.dtype, B: int, N: int, MKBPR: int, SSAR: int):
    test_prefill_kernel(DTYPE, B, N, BLOCK_SIZE_DEFAULT, D_DEFAULT, MKBPR, SSAR)


# --------------------------------------------------------------------------- #
# Test 4 – Large batch
# ---------------------------------------------------------------------------#
DTYPE_LB = [torch.float16, torch.bfloat16]
B_LB = [16, 20, 24, 32]
N_LB = [4, 8, 16]
MKBPR_LB = [1, 64, 126, 128, 130, 141]
SSAR_LB = [True, False]

@pytest.mark.parametrize(
    "DTYPE, B, N, MKBPR, SSAR", product(DTYPE_LB, B_LB, N_LB, MKBPR_LB, SSAR_LB),
    ids=[f"{DTYPE=},{B=},{N=},{MKBPR=},{SSAR=}" for DTYPE, B, N, MKBPR, SSAR in product(DTYPE_LB, B_LB, N_LB, MKBPR_LB, SSAR_LB)]
)
@torch.inference_mode()
def test_large_batch(DTYPE: torch.dtype, B: int, N: int, MKBPR: int, SSAR: int):
    test_prefill_kernel(DTYPE, B, N, BLOCK_SIZE_DEFAULT, D_DEFAULT, MKBPR, SSAR)

# --------------------------------------------------------------------------- #
# Quick manual run (kept for copy-paste debugging)
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    test_prefill_kernel(DTYPE=torch.bfloat16, B=20, N=8, BLOCK_SIZE=128, D=128, MKBPR=128, SSAR=False, verbose=True) # passes 
    print("Manual smoke test PASSED")
