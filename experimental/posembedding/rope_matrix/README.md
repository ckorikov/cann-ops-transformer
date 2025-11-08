Rope-Matrix
Compile and test:

```bash
# How to run this sub-project
rm -rf build_kernel
cmake -B build_kernel .
cmake --build build_kernel

bash build_torch.sh

export LD_LIBRARY_PATH=build_kernel/lib:$LD_LIBRARY_PATH
python ./test_rope.py
```

Support limitation:
```bash
dtype=bf16, x=BNSD, y(matrix)=DD, cos/sin=11SD, D=128.
For example: x = [1, 24, 28800, 128], y = [128, 128], cos/sin = [1, 1, 28800, 128]
```

### Design
As the fig shows, we design a Mathematical Equivalence algorithm "Rope-Matrix(ROME)" to accelerate rope.
The key is to replace tensor rearrage operator by matrix.            
![image.png](https://raw.gitcode.com/user-images/assets/7673863/75dd5e40-a660-4956-b052-95079eae8f8d/image.png 'image.png')
Besides, for 3D-ROPE situation, origin rope-fused need call three times, we just need to call once.
![image.png](https://raw.gitcode.com/user-images/assets/7673863/9a6d1aa0-9e73-4908-a0c7-c3cebfeda407/image.png 'image.png')

Designing detail：
By profiling, we find memory bound for both C and V on 910B. Thus, no need CV pipeline, we just simplify run V after C.
Using ```AscendC::CrossCoreSetFlag``` and ```AscendC::CrossCoreWaitFlag``` to control pipeline.

1. folder design:
```bash
  ${op_class}                                          # class
  ├── ${op_name}                                       # name
  │   ├── inc                                          # define a struct as TCubeTiling, which can be call by both op_device and op_host
  │   │   └── ${op_name}_tiling.h                      
  │   ├── op_host                                      # Tiling、InferShape(If needed, custom operator may not need)...
  │   │   └── ${op_name}_tiling.cpp                    # Tiling
  │   ├── op_kernel                                    # kernel
  │   │   ├── ${op_name}.cpp                           # kernel input
  │   │   ├── ${op_name}.h                             # fused vector part
  │   │   └── ${op_name}_cube.h                        # matrix part
  │   ├── CMakeLists.txt                               # makefile
  │   ├── torch_interface.cpp                          # for torch pybind mode, for torch calling
  │   ├── build_torch.sh                               # 3 file to install a lib which can be call by torch
  │   ├── setup.py
  │   ├── ascendc_extension.py
  │   ├── test_rope.py                                 # test file: contain how to gen the [d, d] rope-matrix
  │   └── README.md                                    # readme
```

2. design host and kernel: take 2 sub-project as reference:
```bash
from example: https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation/baremix_custom.cpp
from example: https://gitcode.com/cann/ops-transformer/blob/master/posembedding/rotary_position_embedding/rotate_half_bf16.h
```
we simplify the code from origin rope-fused operator to fit our cases.
Then use ```ASCEND_IS_AIC, ASCEND_IS_AIV``` to isolation cube/vector process and ```CrossCoreSetFlag, CrossCoreWaitFlag``` to control communication between cube and vector.
```bash
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // dry kernel type: KERNEL_TYPE_MIX_xxx
    TPipe tpipe;
    TCubeTiling tilingLocal;
    RopeMatrix::CopyTiling(&tilingLocal, tiling);
    if ASCEND_IS_AIC {
        ...
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // set flag after matmul
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(0x8); // wait matmul outputs
        ...
    }
```
host tiling can be summary as follows:
```bash
split 'S' with vector core num, the last core may run less than other cores.
For example S=28799, vector_core=40: split to 720*39 + 719:
Then cube process 720*2, vector process 720, last vector process 719, last cube process (720+719)
```

3. fit project to torch. Before fit to torch, you need to compile ```cmake -B build_kernel .``` and ```cmake --build build_kernel```, and can find many auto-gen codes, which you not need to care, just call as follows:
```bash
1). design 'torch_interface.cpp':
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_matrix_kernel_bf16", &rope_matrix_kernel_bf16, "rope_matrix_kernel_bf16");
}
at::Tensor rope_matrix_kernel_bf16(at::Tensor x, at::Tensor y, at::Tensor sin, at::Tensor cos)
{
    ...

    ACLRT_LAUNCH_KERNEL(rope_matrix_kernel_bf16)(
        blockDims, aclstream,
        (uint8_t*)(x.storage().data()),
        (uint8_t*)(y.storage().data()),
        (uint8_t*)(sin.storage().data()),
        (uint8_t*)(cos.storage().data()),
        (uint8_t*)(output.storage().data()),
        (uint8_t*)(workspace_tensor.storage().data()),
        tilingDevice);
}
```
may be you are confused by ```ACLRT_LAUNCH_KERNEL```, this is auto-generated during build, path relate to cmake, current: build_kernel/include/rope_matrix_custom_kernel/.

```bash
2). design compile files.
'build_torch.sh': python setup.py build_ext --inplace
'setup.py': set file path and module name, call common compile file 'ascendc_extension.py'.
'ascendc_extension.py': common compile file, not only can be used by this project, other custom projects can also import this file, library list taken from 'https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/CppExtensions/CMakeLists.txt'
```

4. How to call
```bash
export LD_LIBRARY_PATH=build_kernel/lib:$LD_LIBRARY_PATH
import torch
import torch_npu
import rope_matrix
x = rope_matrix.rope_matrix_kernel_bf16(x, mat, sin, cos)
```