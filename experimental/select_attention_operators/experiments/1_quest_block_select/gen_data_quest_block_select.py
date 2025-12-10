import torch
import torch_npu
import math
from typing import Tuple, List
 
SEED = 42

def gen_quest_inputs(B: int, 
                     H: int,  # number of query heads
                     N: int,  # number of KV heads
                     BLOCK_SIZE: int, 
                     D: int, 
                     device: str,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    produces random matrices for Quest block selection operation:
    - query: [B, H, D] 
    - maxblock: [B, N, BLOCK_SIZE, D]
    - minblock: [B, N, BLOCK_SIZE, D]
    """
    # reset the seed each time to be able to reproduce individual failed tests 
    # out of a loop of tests
    torch.manual_seed(SEED)

    # Generate query tensor [B, H, D]
    query = torch.empty(B, H, D, dtype=dtype).uniform_(-1, 1)
    
    # Generate maxblock tensor [B, N, BLOCK_SIZE, D]
    maxblock = torch.empty(B, N, BLOCK_SIZE, D, dtype=dtype).uniform_(-1, 1)
    
    # Generate minblock tensor [B, N, BLOCK_SIZE, D]
    minblock = torch.empty(B, N, BLOCK_SIZE, D, dtype=dtype).uniform_(-1, 1)
    
    # Move all tensors to device global memory (GM)
    query = query.to(device).contiguous()
    maxblock = maxblock.to(device).contiguous()
    minblock = minblock.to(device).contiguous()
    
    return query, maxblock, minblock


def compare_indices(reference: torch.Tensor, custom: torch.Tensor, tol_percentage:float=0.02, verbose=False) -> bool:
    """
    Compares tensors of integer numbers, requiring the last dimension to contain 
    the same set of numbers.

    Args:
    - reference: Tensor assumed to be correct
    - custom: Tesnor under test
    - compare the last dimension of "reference" vs "custom", disregarding the 
      order (set comparison).
    - tol_percentage - fraction of total elemetns that are allowed to be 
      incorrect. If higher than 0.0, then at least 1 element will be allowed to
      be corrupt.
    - verbose = True <--> print PASSED or FAILED, and specify the number of 
      violations.
    
    Return:
     True <--> reference and custom contain the same (within tolerance)
    """

    sorted_reference = torch.sort(reference, dim=-1).values
    sorted_custom = torch.sort(custom, dim=-1).values

    # Precompute the difference count: number of elements in custom not in reference for each [b, n]
    B, N, k = reference.shape
    diff_count = torch.zeros((B, N), dtype=torch.int32, device=reference.device)

    tol_count = int(math.ceil(B * N * k * tol_percentage))

    # Iterate over each element in the batch and the second dimension
    for b in range(B):
        for n in range(N):
            ref_set = set(reference[b, n].cpu().numpy())
            custom_set = set(custom[b, n].cpu().numpy())
            diff_count[b, n] = len(ref_set - custom_set)
    n_violating_elems = diff_count.sum().sum()
    
    # test summary
    test_ok = n_violating_elems <= tol_count
    if verbose: 
        print(f"{'PASSED' if test_ok else 'FAILED'} - ", end='')
        print(f"{n_violating_elems}/{reference.numel()} indices are incorrect (allowed:{tol_count})")    
    
    return test_ok