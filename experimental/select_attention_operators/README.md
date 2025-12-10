# Select Attention Operators
Ascend-910B kernels for sparse attention. The kernels can be launched through python interface which we provide - see kernel usage examples in the [experiments](experiments) directory.

## Repo structure:

```bash
.
|-- experiments - per kernel: data-generator, reference model, test, benchmark time and bandwidth
|   |-- 1_quest_block_select - simplified predictor (<16k tokens) 
|   |-- 2_quest_prefill_metadata - constructing metadata after prefill
|   |-- 3_quest_block_select_paged - elaborate predictor (unbounded seq. len.)
|-- kernels - python packages, each having one or more ascendc kernels + 1 torch interface
|   |-- select_attn_decoding_ops - predictor kernels
|   `-- select_attn_prefill_ops - metadata construction kernels 
`-- scripts
    |-- build_kernels.sh - builds all kernels
    |-- check_cann.sh - builds all kernels
    |-- init_cann.sh - initialize the enbvironment and Ascend device version
    `-- num_cores_map.sh - mapping "device --> number of cores"
```
## Requirements
Tested to work with:
- Ascend910B2, Ascend910B4
- CANN versions 8.0.RC3.beta1, 8.2.RC2, 8.3.RC1
- Python 3.11.10

## Installation
Create conda environment called "sa" and install teh dependencies
```bash
conda create -n sa python=3.11.10 -y
conda activate sa
pip install -r requirements.txt

```
## Running
Activate conda and CANN environments, compile the operators and build their python api (as python packages).

```bash
source scripts/init_cann.sh Ascend910B4  # change Ascend910B4 to your card model, e.g. Ascend910B2
bash scripts/build_kernels.sh
```

Run all tests that are found in the experiments subdirectory
```bash
pytest -v experiments
```

The kernels are deployed with a very neat built in documentation:
```python
import torch_npu
from select_attn_decoding_ops import quest_block_select_paged
help(quest_block_select_paged)
```
<details>
<summary>Prints:</summary>

```
Help on built-in function quest_block_select_paged in module select_attn_decoding_ops:

quest_block_select_paged(...) method of builtins.PyCapsule instance
    quest_block_select_paged(query: torch.Tensor, maxblocks: torch.Tensor, minblocks: torch.Tensor, metadata_block_tables: torch.Tensor, seq_lens: torch.Tensor, k: int) -> torch.Tensor
    
    
    Interface to the `quest_block_select_paged` kernel which predicts the
    sparsity mask during decoding in the form of top-k important kv-block 
    indices for every KV-head in every request. The returned KV block ids 
    are not the indices in the KV-cache, but rather from their enumeration 
    from 0 to number of blocks in the sequence length being decoded.
    
    Args:
        query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
        maxblocks (torch.Tensor): Quest metadata with maximum vectors of 
                                every key-cache block of shape 
                                [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
        minblocks (torch.Tensor): Quest metadata with minimum vectors of 
                                every key-cache block of shape 
                                [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
        metadata_block_tables (torch.Tensor): Metadata block tables of 
                                            shape [B, MMBPR] (int32)
        seq_lens (torch.Tensor): Sequence length of each request in the batch
                               of shape [B] (int32)
        k (int): Number of highest indices to return for every KV head
    
    Returns:
        torch.Tensor: Selected indices vector of shape [B, N, k] (int32)
    
    Limitations: due to kernel's internal buffer design on 910B:
        D = 128
        BLOCK_SIZE = 128
        H / N <= BLOCK_SIZE
        MMBPR < 7 (below 5 is the most stable)
```


</details>

## Development Workflow for a new kernel "OP"
1. Add new kernel implementations in the `kernels/` directory in one of 2 ways:
   1) under an existing python package e.g. `kernels/select_attn_decoding_ops/`. Then add your kernel code as new OP.cpp, add a compilation line to compile.sh, add a torch interface inside troch_interface.cpp
   2) as a new python package: `kernels/OP/`, with a OP.cpp kernel implementation; torch_interface.cpp, compile.sh, build.sh in it.
2. Create a dedicated experiment directory `experiments/4_OP` and implement in it the following programs:
    - *ref_OP.py* - start off by implementing a reference python model for correctness.
    - *gen_data_OP.py* - a function that produces a set of input tensors for your kernel.
    - *test_OP.py* with a smoke test (single run first) to validate correctness on a focused single input, then extend to automated pytesting across wide range of input shapes/data-types
    - *benchmark_OP.py* - measure performance (time, bandwidth)
