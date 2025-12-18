import torch
import torch_npu
import math
from typing import Tuple, List

SEED = 42

def gen_quest_paged_w_inputs(B: int, 
                           H: int,  # number of query heads
                           N: int,  # number of KV heads
                           BLOCK_SIZE: int, 
                           D: int,
                           num_meta_blocks: int,  # number of metadata blocks to allocate (not all of them will be used. Effectively, only just-enough blocks needed for the sequence length of each request will be actually loaded by the quest_block_select_paged kernel)
                           MMBPR: int,  # Max Metadata Blocks Per Request
                           same_seq_len_all_reqs: bool = True,
                           device: str = 'npu:0',
                           dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Produces random matrices for paged Quest block selection operation:
    - query: [B, H, D] 
    - maxblocks: [num_meta_blocks, BLOCK_SIZE, N, D]
    - minblocks: [num_meta_blocks, BLOCK_SIZE, N, D]
    - metadata_block_tables: [B, MMBPR]
    - seq_lens: [B]
    - tokens_since_metadata_update: int
    """
    assert(num_meta_blocks >= B*MMBPR)

    # reset the seed each time to be able to reproduce individual failed tests 
    # out of a loop of tests
    torch.manual_seed(SEED)

    # Generate query tensor [B, H, D]
    query = torch.empty(B, H, D, dtype=dtype).uniform_(-1, 1)
    
    # Generate maxblocks tensor [num_meta_blocks, BLOCK_SIZE, N, D]
    maxblocks = torch.empty(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype).uniform_(-1, 1)
    
    # Generate minblocks tensor [num_meta_blocks, BLOCK_SIZE, N, D]
    minblocks = torch.empty(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype).uniform_(-1, 1)
    
    # Generate metadata_block_tables [B, MMBPR]
    # Each row represents the metadata block indices for a request
    metadata_block_tables = torch.randint(0, num_meta_blocks, (B, MMBPR), dtype=torch.int32)
    
    # Generate sequence lengths [B]
    # Sequence lengths should be reasonable values (e.g., between 1 and some max length)
    max_seq_len = MMBPR * BLOCK_SIZE * BLOCK_SIZE
    if same_seq_len_all_reqs:
        seq_lens = torch.tensor([max_seq_len]*B, dtype=torch.int32, device=device)
        tokens_since_metadata_update = 0
    else:
        seq_lens = torch.randint(low=0, high=max_seq_len + 1, size=(B,), dtype=torch.int32, device=device)
        tokens_since_metadata_update = torch.randint(0, BLOCK_SIZE, (1,)).item()
    
    # Move all tensors to device global memory (GM)
    query = query.to(device).contiguous()
    maxblocks = maxblocks.to(device).contiguous()
    minblocks = minblocks.to(device).contiguous()
    metadata_block_tables = metadata_block_tables.to(device).contiguous()
    seq_lens = seq_lens.to(device).contiguous()
    
    return query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update


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


def ceil_div(x:int ,y:int) -> int:
    return (x + y - 1) // y