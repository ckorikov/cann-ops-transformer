
## TyphoonMLA

TyphoonMLA is a mixed naive-absorb MLA kernel for shared prefix. For more technical details on TyphoonMLA, check out our preprint [paper](https://arxiv.org/abs/2509.21081).


### Folder structure
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

### Requirements
* CATLASS 1.0
* CANN toolkit 
* CANN-NNAL (required for torch_npu absorb baseline)
* Pytorch & torch_npu

### Build & compile

1. Clone CATLASS
```
git clone https://gitee.com/ascend/catlass.git
cd catlass 
git checkout v1.0.0
export CATLASS_DIR=$(pwd)
cd ..
```

2. Compile kernel and python extension
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