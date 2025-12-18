import torch
from typing import Tuple
from gen_data_quest_prefill_metadata import ceil_div

def ref_quest_prefill_metadata(
        k_cache: torch.Tensor,          # (num_blocks_total, BLOCK_SIZE, N, D)
        block_tables: torch.Tensor,      # (B, max_blocks_per_req)
        seq_lens: torch.Tensor,          # (B,)
        metadata_block_tables: torch.Tensor,  # (B,)
        maxblocks: torch.Tensor,
        minblocks: torch.Tensor
):
    """
    Reference implementation of the QUEST prefill-metadata kernel.
    For every request r (0..B-1) and every KV-head n (0..N-1):
        1.  Iterate over the real KV blocks of that request
            (seq_lens[r] tokens → ⌈seq_lens[r]/BLOCK_SIZE⌉ blocks).
        2.  Per block, load **only head n** of the 4-D K-cache slice
            k_cache[kv_block_id, :, n, :]  → shape (BLOCK_SIZE, D)
        3.  Reduce-max and reduce-min **along the token axis** (dim 0)
            maxblock[meta_block_id, :, n, :] ← max(tokens)   (BLOCK_SIZE → 1 per channel)
            minblock[meta_block_id, :, n, :] ← min(tokens)   (BLOCK_SIZE → 1 per channel)
        4.  Write the two metadata blocks to
            maxblocks[metadata_block_tables[r], :, n, :]
            minblocks[metadata_block_tables[r], :, n, :]

    The function returns the **full** maxblocks/minblocks tensors so you can
    compare them directly against the Ascend-C outputs.
    """
    device = k_cache.device
    dtype = k_cache.dtype
    
    B = block_tables.shape[0]
    BLOCK_SIZE = k_cache.shape[1]
    N = k_cache.shape[2]
    D = k_cache.shape[3]

    MMBPR = metadata_block_tables.shape[1]

    assert(BLOCK_SIZE == 128)
    assert(D == 128)

    # ---- iterate over a single request at a time ----
    for r in range(B):
        num_kv_blocks_in_request = ceil_div(seq_lens[r], BLOCK_SIZE)
        num_meta_blocks_in_request = ceil_div(num_kv_blocks_in_request, BLOCK_SIZE)

        # iterate over BLOCK_SIZE kv blocks tokens of this request to produce one metadata block
        for meta_blk in range(num_meta_blocks_in_request):
            num_kv_blocks_completed = meta_blk * BLOCK_SIZE
            num_kv_blocks_todo_curr_iter = min(num_kv_blocks_in_request - num_kv_blocks_completed, BLOCK_SIZE);            
            meta_blk_id = metadata_block_tables[r, meta_blk].item()
    
            # iterate over one kv block (BLOCK_SIZE tokens)
            for blk in range(num_kv_blocks_todo_curr_iter):
                if (blk == num_kv_blocks_todo_curr_iter - 1) and (meta_blk == num_meta_blocks_in_request - 1):
                    # tail (last KV block) - do not reduce over all tokens!
                    ntokens_reduced_so_far = meta_blk * BLOCK_SIZE * BLOCK_SIZE + blk * BLOCK_SIZE
                    ntokens_to_reduce = seq_lens[r] - ntokens_reduced_so_far
                else:
                    ntokens_to_reduce = BLOCK_SIZE
                kv_block_id = block_tables[r, meta_blk * BLOCK_SIZE + blk].item()   # global block id
                kv_block = k_cache[kv_block_id, :ntokens_to_reduce, :, :]  # (BLOCK_SIZE, N, D)
                maxblocks[meta_blk_id, blk, :, :] = kv_block.max(dim=0)[0]
                minblocks[meta_blk_id, blk, :, :] = kv_block.min(dim=0)[0]

            # tail filling with zeros
            num_unused_metadata_tokens = BLOCK_SIZE - num_kv_blocks_todo_curr_iter
            if (num_unused_metadata_tokens > 0):
                maxblocks[meta_blk_id, num_kv_blocks_todo_curr_iter:, :, :] = 0
                minblocks[meta_blk_id, num_kv_blocks_todo_curr_iter:, :, :] = 0              