"""
Single input testing (for debugging) -> Run this file with python <filename>
Wide range testing (for validation) -> Run this file with pytest <filename>
"""
import pytest
import torch
import torch_npu
from select_attn_decoding_ops import quest_block_select_paged
from ref_quest_block_select_paged import ref_quest_block_select_paged
from gen_data_quest_block_select_paged import gen_quest_paged_inputs, compare_indices


device = "npu:0"
BLOCK_SIZE = 128
D = 128
SAME_SEQ_LEN_ALL_REQS = False


# --------------------------------------------------------------------------- #
# Central test worker  (assertion crashes <--> test failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Skipping direct invocation of the main test_quest_paged_kernel function")
def test_quest_paged_kernel(DTYPE: torch.dtype, B: int, H: int, N: int, BLOCK_SIZE: int, D: int, 
                            MMBPR: int, k: int, verbose: bool = False) -> None:
    """
    Main test function for paged quest kernel
    """
    # Generate input data
    num_meta_blocks = B * MMBPR
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = gen_quest_paged_inputs(
        B, H, N, BLOCK_SIZE, D, num_meta_blocks, MMBPR, SAME_SEQ_LEN_ALL_REQS,
        device, DTYPE
    )

    # Run reference implementation
    ref_ids = ref_quest_block_select_paged(
        query, maxblocks, minblocks, metadata_block_tables, seq_lens, k, fast=True
    )
    
    # Run custom implementation
    custom_ids = quest_block_select_paged(
        query, maxblocks, minblocks, metadata_block_tables, seq_lens, k
    )
    
    # Compare results
    cfg_str = f"{DTYPE=}, {B=}, {H=}, {N=}, {BLOCK_SIZE=}, {D=}, {MMBPR=}, {k=}"
    if verbose:
        print("Input shapes:")
        print(f"  query: {query.shape}")
        print(f"  maxblocks: {maxblocks.shape}")
        print(f"  minblocks: {minblocks.shape}")
        print(f"  metadata_block_tables: {metadata_block_tables.shape}")
        print(f"  seq_lens: {seq_lens.shape}")
        print(f"  Sequence lengths: {seq_lens.tolist()}")
        print(" ========== Reference torch implementation output ========== ")
        print("ids reference:", ref_ids)
        print(ref_ids.shape)       
        print(" ========== Custom ascendc implementation output =========== ")
        print("ids custom:", custom_ids)
        print(custom_ids.shape)
        print(" ========== DIFF =========== ")
        print("ids diff:", ref_ids - custom_ids)
        print(" ==================== SUMMARY =================== ")
        print(cfg_str)
    indices_match = compare_indices(ref_ids, custom_ids, verbose=verbose)               
    
    # Assert all comparisons pass
    assert indices_match, f"Indices comparison failed for {cfg_str}"

def construct_quest_paged_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, MMBPR_VALUES, K_VALUES):
    """
    Construct parameter sets for paged Quest kernel testing with constraints:
    - D, BLOCK_SIZE are fixed
    - N ≤ H
    - H is a multiple of N (H = N * G for natural G)
    """
    parameter_sets = []
    for dtype in DTYPE_VALUES:
        for b in B_VALUES:
            for h in H_VALUES:
                for n in N_VALUES:
                    # Apply constraints: N ≤ H and H is multiple of N
                    if n <= h and h % n == 0:
                        for mmbpr in MMBPR_VALUES:
                            for k in K_VALUES:
                                parameter_sets.append((dtype, b, h, n, BLOCK_SIZE, D, mmbpr, k))
    return parameter_sets

########################### Test 1 - Basic functionality ###########################
DTYPE_VALUES = [torch.float16, torch.bfloat16]
B_VALUES = [1, 2]
H_VALUES = [8, 16]
N_VALUES = [4, 8]
MMBPR_VALUES = [1, 2, 3, 4]
K_VALUES = [4, 8]

parameter_sets = construct_quest_paged_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, MMBPR_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, MMBPR={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]    
)
@torch.inference_mode()
def test_quest_paged_basic(DTYPE:torch.dtype, B: int, H: int, N: int, BLOCK_SIZE: int, D: int, 
                           MMBPR: int, k: int) -> None:
    test_quest_paged_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k)

########################### Test 2 - Edge cases ###########################
DTYPE_VALUES = [torch.float16, torch.bfloat16]
B_VALUES = [1]
H_VALUES = [1, 2, 4, 8, 16, 32]
N_VALUES = [1, 2, 4, 8, 16]
MMBPR_VALUES = [1, 2, 4]
K_VALUES = [1, 4, 16]

parameter_sets = construct_quest_paged_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, MMBPR_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, MMBPR={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)
@torch.inference_mode()
def test_quest_paged_edge_cases(DTYPE:torch.dtype, B: int, H: int, N: int, BLOCK_SIZE: int, D: int, 
                                MMBPR: int, k: int) -> None:
    test_quest_paged_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k)

########################### Test 3 - Extensive testing ###########################
DTYPE_VALUES = [torch.float16, torch.bfloat16]
B_VALUES = [1, 2, 8]
H_VALUES = [8, 16, 32, 64]
N_VALUES = [2, 4, 8, 16]
MMBPR_VALUES = [1, 2, 3, 4]
K_VALUES = [2, 4, 8, 16, 32]

parameter_sets = construct_quest_paged_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, MMBPR_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, MMBPR={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)
@torch.inference_mode()
def test_quest_paged_extensive(DTYPE:torch.dtype, B: int, H: int, N: int, BLOCK_SIZE: int, D: int, 
                              MMBPR: int, k: int) -> None:
    test_quest_paged_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k)

########################### Test 4 - Large scale ###########################
DTYPE_VALUES = [torch.float16, torch.bfloat16]
B_VALUES = [16, 20, 24, 32]
H_VALUES = [32, 64, 128]
N_VALUES = [4, 8, 16, 32, 64]
MMBPR_VALUES = [1, 2, 4]
K_VALUES = [4, 8, 32, 64]

parameter_sets = construct_quest_paged_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, MMBPR_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, MMBPR={mmbpr}, k={k}" 
         for dtype, b, h, n, block_size, d, mmbpr, k in parameter_sets]  
)
@torch.inference_mode()
def test_quest_paged_large_scale(DTYPE:torch.dtype, B: int, H: int, N: int, BLOCK_SIZE: int, 
                                 D: int, MMBPR: int, k: int) -> None:
    test_quest_paged_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, MMBPR, k)


# --------------------------------------------------------------------------- #
# Quick manual run (kept for copy-paste debugging)
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    test_quest_paged_kernel(DTYPE=torch.bfloat16, B=20, H=32, N=8, BLOCK_SIZE=128, D=128, MMBPR=1, k=16, verbose=True)  # fails bfloat16
    print("Manual smoke test PASSED")

