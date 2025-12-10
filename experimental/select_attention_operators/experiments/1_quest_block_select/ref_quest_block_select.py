import itertools
import torch


def ref_quest_block_select(query: torch.Tensor,    # (B,H,D)
              maxblock: torch.Tensor, # (B,N,BLOCK_SIZE,D)
              minblock: torch.Tensor, # (B,N,BLOCK_SIZE,D)
              k: int,
              fast:bool=True) -> torch.Tensor: # (B,N,k)
    """
    Reference implementation of the QUEST block selection operator
    Selects the top-k important KV blocks based on approximate attention scores.

    Input:
        query - fp16 tensor of shape (B,H,D)
        maxblock - fp16 tensor of shape (B,N,BLOCK_SIZE,D) - for every batch,kvhead - the BLOCK_SIZE per-page max-metadata vectors
        minblock - fp16 tensor of shape (B,N,BLOCK_SIZE,D) - for every batch,kvhead - the BLOCK_SIZE per-page mix-metadata vectors
        k - integer specifying how many block-indices to return for each kv-head

    Algorithm steps:
    1. For every batch, reduce-mean the query tensor across H dimension 
        such that every group of H/N vectors of shape D are reduced to one vector
        of shape D denotes as "grouped_query[b,n]".
    2. For each batch b and KV-head n:
        2.1. product_max, product_min = Elementwise-multiply grouped_query[b,n] with each maxblock[b,n] and minblock[b,n] vector
        2.2. channel_max_product = Elementwise-max between the two products (product_min, product_max)
        2.3. block_scores = Reduce-sum the last dimension of approx_attention (D to 1)
        2.4. selected_indices = Find the top-k indices in the last dimension out of the BLOCK_SIZE numbers 
    3. Return selected_indices
    """
    if fast:
        return ref_quest_fast(query, maxblock,minblock, k)
    else:
        return ref_quest_slow(query, maxblock,minblock, k)
    
def ref_quest_slow(query: torch.Tensor,    # (B,H,D)
              maxblock: torch.Tensor,      # (B,N,BLOCK_SIZE,D)
              minblock: torch.Tensor,      # (B,N,BLOCK_SIZE,D)
              k: int) -> torch.Tensor:     # (B,N,k)
    """
    SLOW quest reference functionality - runs slower due to explicit nested 
    loop implementation, but guaranteeing the order of operations
    """        
    B, H, D = query.shape
    N = maxblock.shape[1]
    BLOCK_SIZE = maxblock.shape[2]
    
    # Output tensor for selected indices
    selected_indices = torch.zeros(B, N, k, dtype=torch.int32, device=query.device)
    block_scores = torch.zeros(B, N, BLOCK_SIZE, dtype=torch.float16, device=query.device) 
    
    # Step 1: Reduce query across H dimension to get grouped_query [B, N, D]
    # Assuming H is divisible by N, we reduce every H/N heads to 1
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
    
    # Step 2: Process each block for each batch and head
    for b in range(B):
        for n in range(N):
            # Get current grouped query vector
            current_query = grouped_query[b, n, :]  # Shape: (D,)
            
            # Step 2.1: Elementwise multiply grouped_query with maxblock and minblock
            # Reshape for broadcasting: (D,) * (BLOCK_SIZE, D) -> (BLOCK_SIZE, D)
            product_max = current_query.unsqueeze(0) * maxblock[b, n, :, :]  # (BLOCK_SIZE, D)
            product_min = current_query.unsqueeze(0) * minblock[b, n, :, :]  # (BLOCK_SIZE, D)
            
            # Step 2.2: Elementwise max between product_min and product_max
            channel_max_product = torch.maximum(product_max, product_min)  # (BLOCK_SIZE, D)
            
            # Step 2.3: Reduce sum the last dimension (D to 1)
            block_scores[b,n,:] = torch.sum(channel_max_product, dim=1)  # (BLOCK_SIZE,)
            
            # Step 2.4: Find the top-k indices
            # Get indices that would sort the tensor in descending order
            topk_indices = torch.topk(block_scores[b,n], k, sorted=True)[1]
            
            # Store the selected indices
            selected_indices[b, n, :] = topk_indices.to(torch.int32)            
    
    return selected_indices


def ref_quest_fast(query: torch.Tensor,     # (B,H,D)
                   maxblock: torch.Tensor,  # (B,N,BLOCK_SIZE,D)
                   minblock: torch.Tensor,  # (B,N,BLOCK_SIZE,D)
                   k: int) -> torch.Tensor: # (B,N,k)
    """
    FAST quest reference functionality - runs faster thanks to relying on 
    vectorized functions, but not guaranteeing the order of operations
    """
    B, H, D = query.shape
    N = maxblock.shape[1]
    BLOCK_SIZE = maxblock.shape[2]
    
    # Step 1: Reduce query across H dimension to get grouped_query [B, N, D]
    # Assuming H is divisible by N, we reduce every H/N heads to 1
    heads_per_group = H // N
    
    # Reshape query to [B, N, heads_per_group, D] and take mean along head dimension
    grouped_query = query.view(B, N, heads_per_group, D).mean(dim=2)  # [B, N, D]
    
    # Step 2: Elementwise operations using broadcasting
    # Expand grouped_query to [B, N, 1, D] for broadcasting with [B, N, BLOCK_SIZE, D]
    grouped_query_expanded = grouped_query.unsqueeze(2)  # [B, N, 1, D]
    
    # Step 2.1: Elementwise multiply grouped_query with maxblock and minblock
    product_max = grouped_query_expanded * maxblock  # [B, N, BLOCK_SIZE, D]
    product_min = grouped_query_expanded * minblock  # [B, N, BLOCK_SIZE, D]
    
    # Step 2.2: Elementwise max between product_min and product_max
    channel_max_product = torch.maximum(product_max, product_min)  # [B, N, BLOCK_SIZE, D]
    
    # Step 2.3: Reduce sum the last dimension (D to 1)
    block_scores = torch.sum(channel_max_product, dim=-1)  # [B, N, BLOCK_SIZE]
    
    # Step 2.4: Find the top-k indices
    selected_indices = torch.topk(block_scores, k, dim=-1, sorted=True)[1]
    
    return selected_indices
