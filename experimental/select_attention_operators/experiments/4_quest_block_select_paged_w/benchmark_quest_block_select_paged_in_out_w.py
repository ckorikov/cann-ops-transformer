#!/usr/bin/env python3
# benchmark_quest_block_select_paged.py
"""
Benchmark driver for quest_block_select_paged_in_out_w kernel.

Measures:
*  latency (usec) - NPU timer
*  effective bandwidth (TB/s) - bytes moved / time
*  correctness comparison with reference implementation
"""
import torch
import torch_npu
import itertools

from select_attn_decoding_ops import quest_block_select_paged_in_out_w
from ref_quest_block_select_paged_w import ref_quest_block_select_paged_w
from gen_data_quest_block_select_paged_w import gen_quest_paged_w_inputs, ceil_div, compare_indices


torch.npu.set_device("npu:0")
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
D = 128
SAME_SEQ_LEN_ALL_REQS = True # to set equally long input length and avoid unknown actual size

# --------------------------------------------------------------------------- #
#  bytes-moved calculator
# --------------------------------------------------------------------------- #
def bytes_moved_paged_select(B: int, H: int, N: int, BLOCK_SIZE: int, D: int,
                            MMBPR: int, k: int, seq_lens: torch.Tensor) -> int:
    """
    Global-memory traffic (read + write) for quest_block_select_paged_in_out_w.

    Reads:
    - query: [B, H, D] - B * H * D * 2 bytes (FP16)
    - maxblocks: [?, BLOCK_SIZE, N, D] - num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2 bytes (FP16)
    - minblocks: [?, BLOCK_SIZE, N, D] - num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2 bytes (FP16)
    - metadata_block_tables: [B, MMBPR] - B * MMBPR * 4 bytes (INT32)
    - seq_lens: [B] - B * 4 bytes (INT32)

    Writes:
    - selected_indices: [B, N, k] - B * N * k * 4 bytes (INT32)
    """
    toks_per_meta_block = BLOCK_SIZE * BLOCK_SIZE
    num_effective_metadata_blocks = torch.sum(ceil_div(seq_lens, toks_per_meta_block)).item()

    read_query = B * H * D * 2
    read_maxblocks = num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2
    read_minblocks = num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2
    read_metadata_tables = B * MMBPR * 4
    read_seq_lens = seq_lens.element_size() * seq_lens.numel()
    
    write_indices = B * N * k * 4
    
    return read_query + read_maxblocks + read_minblocks + read_metadata_tables + read_seq_lens + write_indices


# --------------------------------------------------------------------------- #
#  benchmark body
# --------------------------------------------------------------------------- #
def benchmark_quest_block_select_paged():
    run_our = True
    run_ref = True
    n_repeat = 10
    n_warmup = 1
    
    B_vals = [10, 20, 24, 32]
    H_vals = [32]
    N_vals = [8]
    MMBPR_vals = [1, 2, 4, 6]  if torch.bfloat16 else [1, 2, 4, 8, 16]
    # k_vals = [4, 8, 12, 16]
    k_vals = [8, 16, 24, 32]

    if not run_our and not run_ref:
        print("Nothing to run, must set run_our=True or run_ref=True")
        return

    print("=" * 124)
    print(f"  {DTYPE=}  {BLOCK_SIZE=}  {D=}  {SAME_SEQ_LEN_ALL_REQS=}")
    print("=" * 124)
    print(f"{'H':>3} {'N':>3} {'B':>3} {'MMBPR':>6} {'Max_seq_len':>12} {'k':>4} {'Outputs_equal':>15} {'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18} {'Ref_BW_[TB/sec]':>16} {'Our_BW_[TB/sec]':>16}")
    print("-" * 124)

    for b, h, n, mmbpr, k in itertools.product(B_vals, H_vals, N_vals, MMBPR_vals, k_vals):
        
        ######## Check correctness #######
        are_equal = "N/A"
        if run_our and run_ref:
            query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = gen_quest_paged_w_inputs(
                    b, h, n, BLOCK_SIZE, D,
                    num_meta_blocks=b * mmbpr,
                    MMBPR=mmbpr,
                    same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                    device="npu:0", 
                    dtype=DTYPE)
            ref_ids = ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k)
            our_ids = torch.zeros((b, n, k), dtype=torch.int32, device=query.device)
            quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, our_ids)
            tol_percentage = 0.02
            are_equal = compare_indices(ref_ids, our_ids, tol_percentage, verbose=False)
            are_equal = "yes" if are_equal else "no"
        
        ########## Our Implementation ##########
        # Our implementation - generate multiple input sets
        if run_our:        
            input_sets = []
            for i in range(n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = \
                    gen_quest_paged_w_inputs(
                        b, h, n, BLOCK_SIZE, D,
                        num_meta_blocks=b * mmbpr,
                        MMBPR=mmbpr,
                        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                        device="npu:0", 
                        dtype=DTYPE)
                input_sets.append((query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update))
            
            # Our implementation - Warm-up runs
            for i in range(n_warmup):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = input_sets[i]
                if run_our:
                    quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, our_ids)
                if run_ref:
                    ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k)
            torch.npu.synchronize()

            # Our implementation - measurements 
            our_duration = None
            our_bw = None
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            our_ids = torch.zeros((b, n, k), dtype=torch.int32, device=query.device)

            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = input_sets[i]
                quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, our_ids)
            end.record()
            torch.npu.synchronize()
            
            our_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
            total_bytes = bytes_moved_paged_select(b, h, n, BLOCK_SIZE, D, mmbpr, k, seq_lens)
            our_bw = total_bytes / our_duration / 1e6  # TB/s


        ########## Reference ##########
        # Reference implementation - generate multiple input sets
        if run_ref:        
            input_sets = []
            for i in range(n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = \
                    gen_quest_paged_w_inputs(
                        b, h, n, BLOCK_SIZE, D,
                        num_meta_blocks=b * mmbpr,
                        MMBPR=mmbpr,
                        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                        device="npu:0", 
                        dtype=DTYPE)
                input_sets.append((query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update))
            
            # Reference implementation - Warm-up runs
            for i in range(n_warmup):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = input_sets[i]
                ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k)
            torch.npu.synchronize()
            
            # Reference implementation - measurement
            ref_duration = None
            ref_bw = None
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            
            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update = input_sets[i]
                ref_quest_block_select_paged_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, k)
            end.record()
            torch.npu.synchronize()
            
            ref_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
            total_bytes = bytes_moved_paged_select(b, h, n, BLOCK_SIZE, D, mmbpr, k, seq_lens)
            ref_bw = total_bytes / ref_duration / 1e6  # TB/s
        
        ####### Print results #######
        max_seq_len = mmbpr * BLOCK_SIZE * BLOCK_SIZE
        print(f"{h:>3} {n:>3} {b:>3} {mmbpr:>6} {max_seq_len:>12} {k:>4} {are_equal:>15} ", end='')
        
        if run_ref and ref_duration is not None:
            print(f"{ref_duration:>18.2f} ", end='')
        else:
            print(f"{'N/A':>18} ", end='')
        
        if run_our and our_duration is not None:
            print(f"{our_duration:>18.2f} ", end='')
        else:
            print(f"{'N/A':>18} ", end='')
        
        if run_ref and ref_bw is not None:
            print(f"{ref_bw:>16.3f} ", end='')
        else:
            print(f"{'N/A':>16} ", end='')
        
        if run_our and our_bw is not None:
            print(f"{our_bw:>16.3f}")
        else:
            print(f"{'N/A':>16}")

    print("=" * 124)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_quest_block_select_paged()
