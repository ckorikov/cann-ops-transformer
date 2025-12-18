import torch
import math

def ceil_div(a, b):
    """Ceiling division: ceil(a / b)"""
    return -(a // -b)

def ref_quest_block_select_paged_w(query: torch.Tensor,              # (B, H, D)
                                 maxblocks: torch.Tensor,          # (num_meta_blocks, BLOCK_SIZE, N, D)
                                 minblocks: torch.Tensor,          # (num_meta_blocks, BLOCK_SIZE, N, D)
                                 metadata_block_tables: torch.Tensor,  # (B, MMBPR)
                                 seq_lens: torch.Tensor,           # (B)
                                 tokens_since_metadata_update: int,
                                 k: int,
                                 fast: bool = True) -> torch.Tensor:  # (B, N, k)
    """
    Reference implementation of the paged QUEST block selection operator
    Selects the top-k important KV blocks based on approximate attention scores.

    Input:
        query - fp16 tensor of shape (B, H, D)
        maxblocks - fp16 tensor of shape (num_meta_blocks, BLOCK_SIZE, N, D)
        minblocks - fp16 tensor of shape (num_meta_blocks, BLOCK_SIZE, N, D)
        metadata_block_tables - int32 tensor of shape (B, MMBPR)
        seq_lens - int32 tensor of shape (B)
        k - integer specifying how many block-indices to return for each kv-head

    Algorithm steps:
    1. For every batch, reduce-mean the query tensor across H dimension 
        such that every group of H/N vectors of shape D are reduced to one vector
        of shape D denotes as "grouped_query[b,n]".
    2. For each batch b and KV-head n:
        2.1. Determine how many metadata blocks are valid for this request: 
             num_valid_blocks = floor_div(seq_lens[b], BLOCK_SIZE * BLOCK_SIZE)
        2.2. For each valid metadata block:
             - Get the block metadata from maxblocks and minblocks
             - Calculate block scores using the same method as non-paged version
        2.3. Find the top-k indices across all valid blocks
    3. Return selected_indices
    """
    if fast:
        return ref_quest_paged_fast(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k)
    else:
        return ref_quest_paged_slow(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k)


def ref_quest_paged_slow(query: torch.Tensor,              # (B, H, D)
                         maxblocks: torch.Tensor,          # (num_meta_blocks, BLOCK_SIZE, N, D)
                         minblocks: torch.Tensor,          # (num_meta_blocks, BLOCK_SIZE, N, D)
                         metadata_block_tables: torch.Tensor,  # (B, MMBPR)
                         seq_lens: torch.Tensor,           # (B)
                         tokens_since_metadata_update: int,
                         k: int) -> torch.Tensor:          # (B, N, k)
    """
    SLOW paged quest reference functionality - explicit nested loop implementation
    """        
    B, H, D = query.shape
    num_meta_blocks, BLOCK_SIZE, N, D_blocks = maxblocks.shape
    MMBPR = metadata_block_tables.shape[1]
    
    assert D == D_blocks, f"Query dimension {D} doesn't match block dimension {D_blocks}"

    if query.dtype == torch.bfloat16:
        query = query.float()    
    
    # Output tensor for selected indices
    selected_indices = torch.zeros(B, N, k, dtype=torch.int32, device=query.device) - 1
    
    # Step 1: Reduce query across H dimension to get grouped_query [B, N, D]
    heads_per_group = H // N
    grouped_query = torch.zeros(B, N, D, dtype=query.dtype, device=query.device)
    
    for group in range(N):
        # Sum all heads in this group
        group_sum = torch.zeros(B, D, dtype=query.dtype, device=query.device)
        for n in range(heads_per_group):
            head_idx = group * heads_per_group + n
            group_sum += query[:, head_idx, :]
        
        # Average to get grouped query
        grouped_query[:, group, :] = group_sum / heads_per_group
    
    # Step 2: Process each batch and head
    for b in range(B):
        # Determine how many metadata blocks are valid for this request
        num_kv_blocks_with_metadata = (seq_lens[b] - tokens_since_metadata_update) // BLOCK_SIZE # yes, floor divide - assuming we created metadata tokens_since_metadata_update tokens ago, and only for the round  multiple of BLOCK_SIZE tokens 
        num_valid_blocks = ceil_div(num_kv_blocks_with_metadata, BLOCK_SIZE)
        num_valid_blocks = min(num_valid_blocks, MMBPR)  # Cap at MMBPR
        
        # Collect all block scores for this batch across all valid blocks
        all_block_scores = []

        for n in range(N):
            # Get current grouped query vector
            current_query = grouped_query[b, n, :]  # Shape: (D,)
            
            # Process each valid metadata block
            all_scores = torch.zeros(num_valid_blocks * BLOCK_SIZE, dtype=torch.float16, device=query.device)
            
            for block_idx in range(num_valid_blocks):
                # Get the actual metadata block index
                meta_block_id = metadata_block_tables[b, block_idx].item()

                # Get max and min blocks for this metadata block
                maxblock = maxblocks[meta_block_id, :, n, :]  # (BLOCK_SIZE, D)
                minblock = minblocks[meta_block_id, :, n, :]  # (BLOCK_SIZE, D)

                if maxblock.dtype == torch.bfloat16:
                    maxblock = maxblock.float()                
                if minblock.dtype == torch.bfloat16:
                    minblock = minblock.float()      
                     
                
                # Elementwise multiply grouped_query with maxblock and minblock
                product_max = current_query.unsqueeze(0) * maxblock  # (BLOCK_SIZE, D)
                product_min = current_query.unsqueeze(0) * minblock  # (BLOCK_SIZE, D)
                
                # Elementwise max between product_min and product_max
                channel_max_product = torch.maximum(product_max, product_min)  # (BLOCK_SIZE, D)
                
                # Reduce sum the last dimension (D to 1)
                scores = torch.sum(channel_max_product, dim=1)  # (BLOCK_SIZE,)
                
                # Store scores with their global indices
                all_scores[block_idx*BLOCK_SIZE:(block_idx+1)*BLOCK_SIZE] = scores
            
            # add sink
            if (tokens_since_metadata_update >= 0): 
                all_scores[0] = torch.finfo(all_scores.dtype).max

            eff_num_scores = len(all_scores)
            eff_k = min(k, eff_num_scores)            
            selected_indices[b, n, :] = torch.topk(all_scores, eff_k, dim=-1)[1] 

            # Add check whether last index should be added based on not-yet-updated metadata (as indicated by tokens_since_metadata_update and seq_len)
            if (tokens_since_metadata_update >= 0): 
                mru = seq_lens[b] - tokens_since_metadata_update;  # mru = sequence length of this request at the most recent metadata update
                win_size = (mru % BLOCK_SIZE != 0) + (seq_lens[b] // BLOCK_SIZE) - (mru // BLOCK_SIZE); # win_size = number of the most recent KV-blocks in the sequence, which are not yet registered by the  metadata
                # int32_t win_size = ((mru & 0x7f) != 0) + (seq_lens[b] >> 7) - (mru >> 7); # win_size - faster computation version due to statically known fact that BLOCK_SZIE is a powers of two --> 7
                for w in range(1, win_size + 1):
                    selected_indices[b, n, k - w] = ((seq_lens[b] + BLOCK_SIZE - 1) // BLOCK_SIZE) - w

    return selected_indices

def ref_quest_paged_fast(query: torch.Tensor,              # (B, H, D)
                         maxblocks: torch.Tensor,          # (num_meta_blocks, BLOCK_SIZE, N, D)
                         minblocks: torch.Tensor,          # (num_meta_blocks, BLOCK_SIZE, N, D)
                         metadata_block_tables: torch.Tensor,  # (B, MMBPR)
                         seq_lens: torch.Tensor,           # (B)
                         tokens_since_metadata_update: int,
                         k: int) -> torch.Tensor:          # (B, N, k)
    """
    FAST paged quest reference functionality - vectorized implementation
    """
    B, H, D = query.shape
    num_meta_blocks, BLOCK_SIZE, N, D_blocks = maxblocks.shape
    MMBPR = metadata_block_tables.shape[1]
    
    assert D == D_blocks, f"Query dimension {D} doesn't match block dimension {D_blocks}"
    
    # Step 1: Reduce query across H dimension to get grouped_query [B, N, D]
    heads_per_group = H // N

    if query.dtype == torch.bfloat16:
        query = query.float()

    grouped_query = query.view(B, N, heads_per_group, D).mean(dim=2)  # [B, N, D]

    # Output tensor for selected indices
    selected_indices = torch.zeros(B, N, k, dtype=torch.int32, device=query.device) - 1
    
    # Process each batch
    for b in range(B):
        num_valid_blocks = min(ceil_div(seq_lens[b].item(), BLOCK_SIZE * BLOCK_SIZE), MMBPR)
        if num_valid_blocks == 0:
            continue
            
        # Get the metadata block indices for this batch
        meta_block_ids = metadata_block_tables[b, :num_valid_blocks]  # [num_valid_blocks]
        
        # Get grouped query for this batch [N, D]
        batch_query = grouped_query[b]  # [N, D]
        
        # Get all relevant maxblocks and minblocks [num_valid_blocks, BLOCK_SIZE, N, D]
        relevant_maxblocks = maxblocks[meta_block_ids]  # [num_valid_blocks, BLOCK_SIZE, N, D]
        relevant_minblocks = minblocks[meta_block_ids]  # [num_valid_blocks, BLOCK_SIZE, N, D]

        if relevant_maxblocks.dtype == torch.bfloat16:
            relevant_maxblocks = relevant_maxblocks.float()
        if relevant_minblocks.dtype == torch.bfloat16:
            relevant_minblocks = relevant_minblocks.float()

        # Reshape for broadcasting: [N, D] -> [1, 1, N, D]
        batch_query_reshaped = batch_query.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        
        # Elementwise multiply [num_valid_blocks, BLOCK_SIZE, N, D]
        product_max = batch_query_reshaped * relevant_maxblocks
        product_min = batch_query_reshaped * relevant_minblocks
        
        # Elementwise max [num_valid_blocks, BLOCK_SIZE, N, D]
        channel_max_product = torch.maximum(product_max, product_min)
        
        # Reduce sum along D dimension [num_valid_blocks, BLOCK_SIZE, N]
        block_scores = torch.sum(channel_max_product, dim=-1)
        
        # Reshape to combine blocks and BLOCK_SIZE [num_valid_blocks * BLOCK_SIZE, N]
        all_scores = block_scores.permute(2, 0, 1).reshape(N, -1)  # [N, num_valid_blocks * BLOCK_SIZE]

        # add sink
        if (tokens_since_metadata_update >= 0): 
            all_scores[:,0] = torch.finfo(all_scores.dtype).max

        # Get top-k indices from the global indices
        eff_num_scores = all_scores.shape[-1]
        eff_k = min(k, eff_num_scores)    
        selected_indices[b, :, :eff_k] = torch.topk(all_scores, eff_k, dim=-1)[1]

        # Add check whether last index should be added based on not-yet-updated metadata (as indicated by tokens_since_metadata_update and seq_len)
        if (tokens_since_metadata_update >= 0): 
            mru = seq_lens[b] - tokens_since_metadata_update;  # mru = sequence length of this request at the most recent metadata update
            win_size = (mru % BLOCK_SIZE != 0) + (seq_lens[b] // BLOCK_SIZE) - (mru // BLOCK_SIZE); # win_size = number of the most recent KV-blocks in the sequence, which are not yet registered by the  metadata
            # int32_t win_size = ((mru & 0x7f) != 0) + (seq_lens[b] >> 7) - (mru >> 7); # win_size - faster computation version due to statically known fact that BLOCK_SZIE is a powers of two --> 7
            for w in range(1, win_size + 1):
                selected_indices[b, :, k - w] = ((seq_lens[b] + BLOCK_SIZE - 1) // BLOCK_SIZE) - w        
    
    return selected_indices