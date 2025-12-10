"""
gen_data.py  -  synthetic data factory for quest_prefill_metadata
"""
import torch
import torch_npu
from typing import Tuple

SEED = 42


def gen_quest_prefill_inputs(
        B: int, N: int, BLOCK_SIZE: int, D: int,
        num_kv_blocks: int = 100,
        num_meta_blocks: int = 20,
        MKBPR: int = 128,
        MMBPR: int = 1,
        same_seq_len_all_reqs: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "npu:0",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor]:
    """
    Creates pseudo-random inputs for quest_prefill_metadata kernel.
    Arguments:
        B - batch size
        N - number of KV heads
        BLOCK_SIZE - number of tokens per block (equal for metadata block and 
                     kv-cache block)
        D - head dimension
        num_kv_blocks - total number of KV-cache blocks for the entire job (all 
                     requests together)
        num_meta_blocks - total number of metadata blocks for the entire job 
                    (all requests together)
        MKBPR - maximum number of KV-cache blocks per request in a batch    
        MMBPR - maximum number of metadata blocks per request in a batch    
        same_seq_len_all_reqs - True <--> all request in a batch will have the 
                     same sequence length (same number of kv blocks)

    Returns
    -------
        k_cache               : (num_kv_blocks, BLOCK_SIZE, N, D)
        block_tables          : (B, MKBPR) indices into k_cache
        seq_lens              : (B,) between BLOCK_SIZE and BLOCK_SIZE*MKBPR
        metadata_block_tables : (B, MMBPR) indices into [0..num_meta_blocks-1]
        maxblocks             : (num_meta_blocks, BLOCK_SIZE, N, D) indices in 
                                [0..num_meta_blocks-1]
        minblocks             : (num_meta_blocks, BLOCK_SIZE, N, D) indices in 
                                [0..num_meta_blocks-1]
    """

    device = torch.device(device)
    
    # reset the seed each time to be able to reproduce individual failed tests 
    # out of a loop of tests
    torch.manual_seed(SEED)    

    # ---- K-cache ---- #
    k_cache = torch.randn(num_kv_blocks, BLOCK_SIZE, N, D,
                         dtype=dtype, device=device) * 1.5 # 1.5 to increase the range

    # ---- request statistics ---- #
    max_seq_len = MKBPR * BLOCK_SIZE
    if same_seq_len_all_reqs:
        seq_lens = torch.tensor([max_seq_len]*B, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.randint(low=0, high=max_seq_len + 1, size=(B,), dtype=torch.int32, device=device)

    # ---- block tables ---- #
    perm_kv_blk_ids = torch.randperm(num_kv_blocks, device=device)[:num_kv_blocks]      
    block_tables = perm_kv_blk_ids.reshape((B, MKBPR)).to(dtype=torch.int32, device=device)                       

    # ---- metadata_block_tables ---- #
    perm_meta_blk_ids = torch.randperm(num_meta_blocks, device=device)[:num_meta_blocks]      
    metadata_block_tables = perm_meta_blk_ids.reshape((B, MMBPR)).to(dtype=torch.int32, device=device)      

    # placeholder for kernel outputs
    maxblocks = torch.empty(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype, device=device)
    minblocks = torch.empty(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype, device=device)

    # Ensure all tensors are contiguous
    k_cache = k_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()
    metadata_block_tables = metadata_block_tables.contiguous()
    maxblocks = maxblocks.contiguous()
    minblocks = minblocks.contiguous()

    return k_cache, block_tables, seq_lens, metadata_block_tables, maxblocks, minblocks


def compare_tensors(ref: torch.Tensor, custom: torch.Tensor, *, rtol=1e-2, atol=1e-3, verbose:bool=True) -> bool:
    """
    compare tensors with a relaxed fp16 tolerance
    """
    if ref.shape != custom.shape:
        print(f"ERROR: shape mismatch  ref={ref.shape}  custom={custom.shape}")
        return False
    try:
        torch.testing.assert_close(ref, custom, rtol=rtol, atol=atol)
        if verbose: print("PASSED")
        return True
    except AssertionError as e:
        if verbose: print("FAILED")
        print(e)
        return False

def ceil_div(x:int ,y:int) -> int:
    return (x + y - 1) // y