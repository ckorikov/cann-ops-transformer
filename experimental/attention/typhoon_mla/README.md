
## TyphoonMLA

TyphoonMLA is a mixed naive-absorb MLA kernel for shared prefix. For more technical details on TyphoonMLA, check out our preprint [paper](https://arxiv.org/abs/2509.21081).


### Folder structure
```
typhoon_mla/
├── src/                    # Source code
│   ├── python_extension/   # Python bindings
│   ├── shared_lib/         # CATLASS kernel impl.
│   └── typhoon_mla.py/     # Python wrappers
│
├── tests/                  # Unit tests
│
├── docs/                   # Documentation
│
├── build/                  # Build output (ignored in VCS)
│
├── bench.py                # Perf. benchmark
├── example.py              # Example usage
├── setup.py                # Setup for python package
├── README.md
└── .gitignore
```

### Requirements
* CATLASS v1.0.0
* CANN toolkit 
* CANN-NNAL (required for torch_npu absorb baseline)
* Torch & torch_npu

### Build & compile

1. Clone CATLASS
```
git clone https://gitee.com/ascend/catlass.git
cd catlass 
git checkout v1.0.0
export CATLASS_DIR=$(pwd)
cd ..
```

2. Set CANN environment
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/driver/bin/setenv.bash 
source /usr/local/Ascend/nnal/atb/set_env.sh # Required for the torch_npu absorb baseline
```

3. Compile kernel and python extension
```
cd src
bash install.sh
cd ..
```

### Run TyphoonMLA kernel
```
python example.py 
```

### Verify functional correctness
```
pytest tests
```

### Performance benchmark
```
python bench.py
```
Following benchmark results are obtained in an Ascend 910B2 NPU:
```
bsz: 64    shared_kv_seqlen: 4096  nonshared_kv_seqlen: 128   | TyphoonMLA (TBT): 1.70 ms   TorchNPU-Absorb (TBT): 1.32 ms
bsz: 128   shared_kv_seqlen: 4096  nonshared_kv_seqlen: 128   | TyphoonMLA (TBT): 1.78 ms   TorchNPU-Absorb (TBT): 2.00 ms
bsz: 256   shared_kv_seqlen: 4096  nonshared_kv_seqlen: 128   | TyphoonMLA (TBT): 2.14 ms   TorchNPU-Absorb (TBT): 3.44 ms
bsz: 512   shared_kv_seqlen: 4096  nonshared_kv_seqlen: 128   | TyphoonMLA (TBT): 3.21 ms   TorchNPU-Absorb (TBT): 6.20 ms
```


### Tested on
```
Ascend 910B2
driver: 25.6.rc1.b010
CANN: 8.2.RC1.alpha002
Python: 3.11.10
torch: 2.6.0
torch_npu: 2.6.0rc1
OS: ubuntu: 22.04
```