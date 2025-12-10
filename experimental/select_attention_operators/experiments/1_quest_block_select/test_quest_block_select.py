"""
Single input testing (for debugging) -> Run this file with python <filename>
Wide range testing (for validation) -> Run this file with pytest <filename>
"""
import pytest
import torch
import torch_npu
from select_attn_decoding_ops import quest_block_select
from ref_quest_block_select import ref_quest_block_select
from gen_data_quest_block_select import gen_quest_inputs, compare_indices


device = "npu:0"   # must have a valid device ID
BLOCK_SIZE=128
D=128


# --------------------------------------------------------------------------- #
# Central test worker  (assertion crashes <--> test failed)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="Skipping direct invocation of the main test_quest_kernel function")
def test_quest_kernel(DTYPE: torch.dtype, B:int, H:int, N:int, BLOCK_SIZE:int, D:int, k:int, verbose:bool = False) -> None:
    """
    Main test function for quest kernel
    """
    # Generate input data
    query, maxblock, minblock = gen_quest_inputs(B, H, N, BLOCK_SIZE, D, device, DTYPE)

    # Run reference implementation
    ref_ids = ref_quest_block_select(query, maxblock, minblock, k)
    
    # Run custom implementation
    custom_ids = quest_block_select(query, maxblock, minblock, k)
        
    # Compare results
    cfg_str = f"{DTYPE=}, {B=}, {H=}, {N=}, {BLOCK_SIZE=}, {D=}, {k=}"
    if verbose:
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
    assert indices_match, f"Indices comparison failed {cfg_str}"

@pytest.mark.skip(reason="Skipping direct invocation of the construct_quest_parameter_sets function")
def construct_quest_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, K_VALUES):
    """
    Construct parameter sets for Quest kernel testing with constraints:
    - D, BLOCK_SIZE are (fixed)
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
                        for k in K_VALUES:
                            parameter_sets.append((dtype, b, h, n, BLOCK_SIZE, D, k))
    return parameter_sets

########################### Test 1 - Basic functionality ###########################
DTYPE_VALUES = [torch.float16]  # torch.bfloat16 is not supported
B_VALUES = [1, 2]
H_VALUES = [8, 16]  # Multiples of N values
N_VALUES = [4, 8]   # Divisors of H values
K_VALUES = [4, 8]

parameter_sets = construct_quest_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, k={k}" 
         for dtype, b, h, n, block_size, d, k in parameter_sets]
)
@torch.inference_mode()
def test_quest_basic(DTYPE:torch.dtype, B:int, H:int, N:int, BLOCK_SIZE:int, D:int, k:int) -> None:
    test_quest_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, k)

########################### Test 2 - Edge cases ###########################
DTYPE_VALUES = [torch.float16]  # torch.bfloat16 is not supported
B_VALUES = [1]
H_VALUES = [1, 2, 4, 8, 16, 32]  # Various multiples
N_VALUES = [1, 2, 4, 8, 16]      # Various divisors
K_VALUES = [1, 4, 16]

parameter_sets = construct_quest_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, k={k}" 
         for dtype, b, h, n, block_size, d, k in parameter_sets]
)
@torch.inference_mode()
def test_quest_edge_cases(DTYPE:torch.dtype, B:int, H:int, N:int, BLOCK_SIZE:int, D:int, k:int) -> None:
    test_quest_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, k)

########################### Test 3 - Extensive testing ###########################
DTYPE_VALUES = [torch.float16]  # torch.bfloat16 is not supported
B_VALUES = [1, 2, 4, 8]
H_VALUES = [8, 16, 32, 64]  # Multiples of various N values
N_VALUES = [2, 4, 8, 16]    # Divisors of various H values
K_VALUES = [2, 4, 8, 16, 32]

parameter_sets = construct_quest_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, k={k}" 
         for dtype, b, h, n, block_size, d, k in parameter_sets]
)
@torch.inference_mode()
def test_quest_extensive(DTYPE:torch.dtype, B:int, H:int, N:int, BLOCK_SIZE:int, D:int, k:int) -> None:
    test_quest_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, k)

########################### Test 4 - Large scale ###########################
DTYPE_VALUES = [torch.float16]  # torch.bfloat16 is not supported
B_VALUES = [16, 20, 21, 22, 23, 24, 25, 32]
H_VALUES = [32, 64, 128]    # Large H values
N_VALUES = [4, 8, 16, 32, 64] # Large N values that divide H
K_VALUES = [4, 8, 32, 64]

parameter_sets = construct_quest_parameter_sets(DTYPE_VALUES, B_VALUES, H_VALUES, N_VALUES, K_VALUES)

@pytest.mark.parametrize(
    "DTYPE, B, H, N, BLOCK_SIZE, D, k", parameter_sets,
    ids=[f"DTYPE={dtype}, B={b}, H={h}, N={n}, BLOCK_SIZE={block_size}, D={d}, k={k}" 
         for dtype, b, h, n, block_size, d, k in parameter_sets]
)
@torch.inference_mode()
def test_quest_large_scale(DTYPE:torch.dtype, B:int, H:int, N:int, BLOCK_SIZE:int, D:int, k:int) -> None:
    test_quest_kernel(DTYPE, B, H, N, BLOCK_SIZE, D, k)


# --------------------------------------------------------------------------- #
# Quick manual run (kept for copy-paste debugging)
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    test_quest_kernel(DTYPE=torch.bfloat16, B=2, H=8, N=4, BLOCK_SIZE=128, D=128, k=4, verbose=True) # currently fails for bfloat
    print("Manual smoke test PASSED")
