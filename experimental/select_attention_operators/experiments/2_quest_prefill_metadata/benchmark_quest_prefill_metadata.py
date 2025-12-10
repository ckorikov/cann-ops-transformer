#!/usr/bin/env python3
# benchmark_quest_prefill_metadata.py
"""
Benchmark driver for quest_prefill_metadata kernel (Ascend-C vs reference).

Measures:
*  latency (μs) – NPU timer
*  effective bandwidth (TB/s) – bytes moved / time
*  correctness comparison with reference implementation
"""
import torch
import torch_npu
import itertools

from select_attn_prefill_ops import quest_prefill_metadata
from ref_quest_prefill_metadata import ref_quest_prefill_metadata
from gen_data_quest_prefill_metadata import gen_quest_prefill_inputs, ceil_div, compare_tensors


torch.npu.set_device("npu:0")
DTYPE = torch.bfloat16
BLOCK_SIZE = 128
D = 128
SAME_SEQ_LEN_ALL_REQS = True # to set equally long input length and avoid unknown actual size

# --------------------------------------------------------------------------- #
#  bytes-moved calculator
# --------------------------------------------------------------------------- #
def bytes_moved_prefill(B: int, N: int, BLOCK_SIZE: int, D: int,
                        MKBPR: int, MMBPR: int, 
                        seq_lens: torch.Tensor) -> int:
    """
    Global-memory traffic (read + write) for quest_prefill_metadata.

    Arguments:
        B - batch size
        N - number of KV heads
        BLOCK_SIZE - number of tokens per block (equal for metadata block and 
                     kv-cache block)
        D - head dimension
        MKBPR - maximum number of KV-cache blocks per request in a batch
        MMBPR - maximum number of metadata blocks per request in a batch

    Reads
    -----
    seq_lens              :  B * 4                                              (int32)
    k_cache               :  num_effective_kv_blocks * BLOCK_SIZE * N * D * 2   (fp16)
    block_tables          :  num_effective_kv_blocks * 4                        (int32)
    metadata_block_tables :  num_effective_kv_blocks * 4                        (int32)

    Writes
    ------
    maxblocks      :  num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2    (fp16)
    minblocks      :  num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2    (fp16)

    Where num_effective_kv_blocks is the total number of K blocks effectively being read 
    by the entire job (all requests) and num_effective_metadata_blocks is the effective 
    number of metadata blocks being produced and written back to GM by the entire kernel
    """

    # total number of K blocks that will be read in the entire job (all requests)
    num_effective_kv_blocks = torch.sum(ceil_div(seq_lens, BLOCK_SIZE)).item()
    
    # total number of K blocks that will be read in the entire job (all requests)
    toks_per_meta_block = BLOCK_SIZE * BLOCK_SIZE
    num_effective_metadata_blocks = torch.sum(ceil_div(seq_lens, toks_per_meta_block)).item()

    read_k   = num_effective_kv_blocks * BLOCK_SIZE * N * D * 2
    read_tbl = num_effective_kv_blocks * 4
    write_meta = 2 * (num_effective_metadata_blocks * BLOCK_SIZE * N * D * 2)
    return read_k + read_tbl + write_meta


# --------------------------------------------------------------------------- #
#  benchmark body
# --------------------------------------------------------------------------- #
def benchmark_quest_prefill():
    run_our = True
    run_ref = True
    n_repeat = 6
    n_warmup = 1
    
    B_vals     = [10, 20, 24, 32]
    N_vals     = [8]
    MKBPR_vals = [63, 80, 94, 128, 150, 200, 256] # max_kv_blocks_per_request
    
    if not run_our and not run_ref:
        print("Nothing to run, must set run_our=True or run_ref=True")
        return

    print("=" * 106)
    print(f"  {DTYPE=}  {BLOCK_SIZE=}  {D=}  {SAME_SEQ_LEN_ALL_REQS=}")
    print("=" * 106)
    print(f"{'N':>3} {'B':>3} {'Seq_len':>10} {'Outputs_equal':>15} {'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18} {'Ref_BW_[TB/sec]':>16} {'Our_BW_[TB/sec]':>16}")
    print("-" * 106)

    for n, b, mkbpr in itertools.product(N_vals, B_vals, MKBPR_vals):
        seq_len = mkbpr * BLOCK_SIZE
        mmbpr = ceil_div(mkbpr, BLOCK_SIZE)
        
        ######## Check correctness ########
        are_equal = "N/A"
        if run_our and run_ref:
            # Create fresh output tensors for correctness check
            k_cache, block_tables, seq_lens, metadata_block_tables, max_out_our, min_out_our = gen_quest_prefill_inputs(
                b, n, BLOCK_SIZE, D,
                num_kv_blocks=b * mkbpr,
                num_meta_blocks=b * mmbpr,
                MKBPR=mkbpr,
                MMBPR=mmbpr,
                same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                device="npu:0", 
                dtype=DTYPE)
            max_out_ref = max_out_our.clone()
            min_out_ref = min_out_our.clone()
            # Run both implementations
            ref_quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out_ref, min_out_ref)
            quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out_our, min_out_our)
            
            if compare_tensors(max_out_ref, max_out_our, verbose=False) and compare_tensors(min_out_ref, min_out_our, verbose=False):
                are_equal = "yes"
            else:
                are_equal = "no"

        ############ Our implementation ###########
        if run_our:
            # Our implementation - gen data (multiple different input sets to prevent caching)
            input_sets = []
            for i in range(n_warmup + n_repeat):
                k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = \
                    gen_quest_prefill_inputs(
                        b, n, BLOCK_SIZE, D,
                        num_kv_blocks=b * mkbpr,
                        num_meta_blocks=b * mmbpr,
                        MKBPR=mkbpr,
                        MMBPR=mmbpr,
                        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                        device="npu:0", 
                        dtype=DTYPE)
                input_sets.append((k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out))

            # Our implementation - Warm-up runs
            for i in range(n_warmup):
                k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
                quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)        
            torch.npu.synchronize()

            # Our implementation - measure time
            our_duration = None
            our_bw = None
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            
            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
                quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)
            end.record()
            torch.npu.synchronize()
            
            our_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
            total_bytes = bytes_moved_prefill(b, n, BLOCK_SIZE, D, mkbpr, mmbpr, seq_lens)
            our_bw = total_bytes / our_duration / 1e6  # TB/s

        ############ Reference ###########
        # Reference implementation - gen data (multiple different input sets to prevent caching)
        if run_ref:
            input_sets = []
            for i in range(n_warmup + n_repeat):
                k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = \
                    gen_quest_prefill_inputs(
                        b, n, BLOCK_SIZE, D,
                        num_kv_blocks=b * mkbpr,
                        num_meta_blocks=b * mmbpr,
                        MKBPR=mkbpr,
                        MMBPR=mmbpr,
                        same_seq_len_all_reqs=SAME_SEQ_LEN_ALL_REQS,
                        device="npu:0", 
                        dtype=DTYPE)
                input_sets.append((k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out))

            # Reference implementation - Warm-up runs
            for i in range(n_warmup):
                k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
                ref_quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)        
            torch.npu.synchronize()

            # Reference implementation - measure time
            ref_duration = None
            ref_bw = None
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            
            start.record()
            for i in range(n_warmup, n_warmup + n_repeat):
                k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out = input_sets[i]
                ref_quest_prefill_metadata(k_cache, block_tables, seq_lens, metadata_block_tables, max_out, min_out)
            end.record()
            torch.npu.synchronize()
            
            ref_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
            total_bytes = bytes_moved_prefill(b, n, BLOCK_SIZE, D, mkbpr, mmbpr, seq_lens)
            ref_bw = total_bytes / ref_duration / 1e6  # TB/s
        
        ####### Print results #######
        print(f"{n:>3} {b:>3} {seq_len:>10} {are_equal:>15} ", end='')
        
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

    print("=" * 106)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_quest_prefill()
