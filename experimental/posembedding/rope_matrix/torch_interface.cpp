#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "acl/acl.h"
#include "aclrtlaunch_rope_matrix_kernel_bf16.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "rope_matrix_tiling.h"

extern void run_rope_matrix_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* sin, uint8_t* cos, uint8_t* out, uint8_t* workspace, uint8_t* tiling);
extern uint8_t *GenerateTiling(RopeMatrixTiling *ropeTiling);

void PackInputInfo(at::Tensor &x, uint32_t blockDim, RopeMatrixTiling *ropeTiling)
{
    ropeTiling->blockDim = blockDim;
    ropeTiling->b = x.sizes()[0];
    ropeTiling->n = x.sizes()[1];
    ropeTiling->s = x.sizes()[2];
    ropeTiling->d = x.sizes()[3];
    return;
}

at::Tensor rope_matrix_kernel_bf16(at::Tensor x, at::Tensor y, at::Tensor sin, at::Tensor cos){
    uint32_t blockDims = 20;
    //only support bnsd
    RopeMatrixTiling ropeTiling;
    PackInputInfo(x, blockDims, &ropeTiling);
    uint64_t len = x.numel();
    // set device
    int devidx = x.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    // output if use "at::Tensor output = torch::zeros(x.sizes(), x.options());", will have extra cost
    at::Tensor output = torch::empty(x.sizes(), x.options());

    // workspace
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto user_workspace_size = len * 2; // need when need copy out and copy in tmp; x.numel() * sizeof(bf16)
    size_t system_workspace_size = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    size_t workspace_size = user_workspace_size + system_workspace_size;
    auto workspace_tensor = 
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));
    
    // host tiling
    size_t cTilingFileSize = sizeof(TCubeTiling);
    size_t vTilingFileSize = sizeof(RopeMatrixTiling);
    size_t tilingFileSize = cTilingFileSize + vTilingFileSize;
    uint8_t *tilingHost;
    uint8_t *tilingDevice;

    aclrtMallocHost((void **)(&tilingHost), tilingFileSize);
    aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingHost, cTilingFileSize, GenerateTiling(&ropeTiling), cTilingFileSize, ACL_MEMCPY_HOST_TO_HOST);
    aclrtMemcpy(tilingHost + cTilingFileSize, vTilingFileSize, (uint8_t*)(&ropeTiling), vTilingFileSize, ACL_MEMCPY_HOST_TO_HOST);
    aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // run : uint8_t *xPtr = reinterpret_cast<uint8_t *>(x.storage().data_ptr().get()); get ptr
    ACLRT_LAUNCH_KERNEL(rope_matrix_kernel_bf16)(
        blockDims, aclstream,
        (uint8_t*)(x.storage().data()),
        (uint8_t*)(y.storage().data()),
        (uint8_t*)(sin.storage().data()),
        (uint8_t*)(cos.storage().data()),
        (uint8_t*)(output.storage().data()),
        (uint8_t*)(workspace_tensor.storage().data()),
        tilingDevice);
    
    // memory clean-up
    aclrtFreeHost(tilingHost);
    aclrtFree(tilingDevice);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_matrix_kernel_bf16", &rope_matrix_kernel_bf16, "rope_matrix_kernel_bf16");
}