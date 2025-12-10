import torch
import torch_npu
from gen_data_quest_block_select import gen_quest_inputs, compare_indices
from select_attn_decoding_ops import quest_block_select
from ref_quest_block_select import ref_quest_block_select


torch.npu.set_device("npu:0")  # make sure having a valid device ID
DTYPE = torch.bfloat16

def compute_total_moved_bytes(B:int, H:int, N:int, BLOCK_SIZE:int, D:int, k:int) -> int:
    """
    Compute the number of bytes transferred between the global memory and the 
    core (read + write) for the Quest operation.

    Reads:
    - query: [B, H, D] - B * H * D * 2 bytes (FP16)
    - maxblock: [B, N, BLOCK_SIZE, D] - B * N * BLOCK_SIZE * D * 2 bytes (FP16)
    - minblock: [B, N, BLOCK_SIZE, D] - B * N * BLOCK_SIZE * D * 2 bytes (FP16)

    Writes:
    - selected_indices: [B, N, k] - B * N * k * 4 bytes (INT32)
    """
    bytes_read_query = B * H * D * 2  # FP16 -> Bytes
    bytes_read_maxblock = B * N * BLOCK_SIZE * D * 2  # FP16 -> Bytes
    bytes_read_minblock = B * N * BLOCK_SIZE * D * 2  # FP16 -> Bytes
    bytes_read = bytes_read_query + bytes_read_maxblock + bytes_read_minblock
    
    bytes_write_indices = B * N * k * 4  # INT32 -> Bytes (indices)
    bytes_write_values = B * N * k * 2  # fp16 -> Bytes (values)
    bytes_write = bytes_write_indices + bytes_write_values
    
    bytes_total = bytes_read + bytes_write  # read + write
    return bytes_total


def benchmark_quest_block_select():
    run_our = True
    run_ref = True
    n_repeat = 10
    n_warmup = 1
    
    # Benchmark parameters
    B_list = [10, 20, 24, 32]
    H = 32
    N = 8
    BLOCK_SIZE = 128
    D = 128
    k_list = [4, 8, 12, 16]
    
    print("=" * 118)
    print(f"  {DTYPE=}  {BLOCK_SIZE=}  {D=}")
    print("=" * 118)
    print(f"{'H':>3} {'N':>3} {'B':>3} {'k':>4} {'Outputs_equal_within_2%':>24} {'Torch_Latency_[usec]':>20} {'Our_Latency_[usec]':>19} {'Torch_BW_[TB/sec]':>18} {'Our_BW_[TB/sec]':>16}")
    print("-" * 118)
    
    for B in B_list:
        for k in k_list:
            # Generate data for benchmarking
            x_list = [gen_quest_inputs(B=B, H=H, N=N, BLOCK_SIZE=BLOCK_SIZE, D=D, device="npu:0", dtype=DTYPE) 
                     for _ in range(n_warmup + 2 * n_repeat)]
            
            # Warm-up runs
            for i in range(n_warmup):
                x = x_list[i]
                if run_our:
                    quest_block_select(*x, k)
                if run_ref:
                    ref_quest_block_select(*x, k)
            
            torch.npu.synchronize()
            
            # Check correctness
            are_equal = "N/A"
            if run_our and run_ref:
                x_test = x_list[-1]
                ref_ids = ref_quest_block_select(*x_test, k)
                our_ids = quest_block_select(*x_test, k)
                tol_percentage = 0.02
                are_equal = compare_indices(ref_ids, our_ids, tol_percentage, verbose=False)
                are_equal = "yes" if are_equal else "no"
            
            # Benchmark our implementation
            our_duration = None
            our_bw = None
            if run_our:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                
                start.record()
                for i in range(n_warmup, n_warmup + n_repeat):
                    quest_block_select(*x_list[i], k)
                end.record()
                torch.npu.synchronize()
                
                our_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
                total_bytes = compute_total_moved_bytes(B, H, N, BLOCK_SIZE, D, k)
                our_bw = total_bytes / our_duration / 1e6  # TB/s
            
            # Benchmark reference implementation
            ref_duration = None
            ref_bw = None
            if run_ref:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                
                start.record()
                for i in range(n_warmup + n_repeat, n_warmup + 2 * n_repeat):
                    ref_quest_block_select(*x_list[i], k)
                end.record()
                torch.npu.synchronize()
                
                ref_duration = start.elapsed_time(end) / n_repeat * 1000  # ms to μs
                total_bytes = compute_total_moved_bytes(B, H, N, BLOCK_SIZE, D, k)
                ref_bw = total_bytes / ref_duration / 1e6  # TB/s
            
            
            # Print row
            print(f"{H:3d} {N:3d} {B:3d} {k:4d} {are_equal:>24} ", end='')
            if ref_duration is not None:
                print(f"{ref_duration:>20.2f} ", end='')
            else:
                print(f"{'N/A':>20} ", end='')
            if our_duration is not None:
                print(f"{our_duration:>19.2f} ", end='')
            else:
                print(f"{'N/A':>19} ", end='')
            if ref_bw is not None:
                print(f"{ref_bw:>18.3f} ", end='')
            else:
                print(f"{'N/A':>18} ", end='')
            if our_bw is not None:
                print(f"{our_bw:>16.3f}")
            else:
                print(f"{'N/A':>16}")
    
    print("=" * 118)


if __name__ == "__main__":
    benchmark_quest_block_select()
