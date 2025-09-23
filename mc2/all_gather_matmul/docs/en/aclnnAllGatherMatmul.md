# aclnnAllGatherMatmul

## Supported Products
- Ascend 910B AI Processor
- Ascend 910_93 AI Processor

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Prototype

Each operator has [two-phase API](common/two_phase_api.md) calls. First, aclnnAllGatherMatmulGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, aclnnAllGatherMatmul is called to perform computation.

*  `aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const char *group, int64_t gatherIndex, int64_t commTurn, int64_t streamMode, const aclTensor *output, const aclTensor *gatherOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
*  `aclnnStatus aclnnAllGatherMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## Function

-   Description: Implements all\_gather communication and matmul fusion.
-   **Formula**:

    $$
    output=allgather(x1)@x2+bias
    $$
    $$
    gatherOut=allgather(x1)
    $$

## aclnnAllGatherMatmulGetWorkspaceSize

-   **Parameters**:
    -   x1 (aclTensor\*, compute input): aclTensor on the device, that is, x1 in the formula. The data type can be FLOAT16 or BFLOAT16, and must be the same as that of x2. The [data format](common/data_format.md) can be ND. **The current version supports only two-dimensional input and non-transposition.**
    -   x2 (aclTensor\*, compute input): aclTensor on the device, that is, x2 in the formula. The data type can be FLOAT16 or BFLOAT16, and must be the same as that of x1. The [data format](common/data_format.md) can be ND. The [non-contiguous tensors](common/non_contiguous_tensors.md) constructed by transposing are supported. **The current version supports only 2-dimensional inputs.**
    -   bias (aclTensor\*, compute input): aclTensor on the device, that is, bias in the formula. The data type can be FLOAT16 or BFLOAT16. The [data format](common/data_format.md) can be ND. Null pointers can be passed. **The current version supports only one-dimensional input and does not support the scenario where bias is not 0.**
    -   group (char\*, compute input): string on the host, indicating the communicator name. The data type can be string. It is obtained through the Hccl API extern HcclResult HcclGetCommName(HcclComm comm, char* commName);. commName is group.
    -   gatherIndex (int64\_t, compute input): integer on the host, indicating the gather target. The value 0 indicates that the target is x1, and the value 1 indicates that the target is x2. The data type can be INT64. The current version supports only **0**.
    -   commTurn (int64\_t, compute input): integer on the host, number of split communication data copies, equal to the value of total data volume divided by single communication volume. The data type can be INT64. The current version supports only **0**.
    -   streamMode (int64\_t, compute input): integer on the host, indicating the enumeration of the AscendCL stream mode. Currently, only the enumerated value 1 is supported. The data type can be INT64.
    -   output (aclTensor\*, compute output): aclTensor on the device, indicating the result of all\_gather communication and mm computation, that is, output in the formula. The data type can be FLOAT16 or BFLOAT16, and must be the same as that of x1. The [data format](common/data_format.md) can be ND.
    -   gatherOut (aclTensor\*, output): aclTensor on the device. Only the result after all_gather communication is output, that is, gatherOut in the calculation formula. The data type can be FLOAT16 or BFLOAT16, and must be the same as that of x1. The [data format](common/data_format.md) can be ND.
    -   workspaceSize (uint64\_t\*, output): size of the workspace to be allocated on the device.
    -   executor (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

-   **Returns**:

    aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

    ```
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input x1, x2, or output is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of x1, x2, bias, or output is not supported.
                                      2. streamMode is not within the valid range.
                                      3. x1 is an empty tensor.
    ```

## aclnnAllGatherMatmul

-   **Parameters**:
    -   workspace (void\*, input): address of the workspace to be allocated on the device.
    -   workspaceSize (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling aclnnAllGatherMatmulGetWorkspaceSize.
    -   executor (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    -   stream (aclrtStream, input): AscendCL stream for executing the task.

-   **Returns**:

    aclnnStatus: status code. For details, see [aclnn Return Codes](common/aclnn_return_codes.md).

## Constraints

- Input x1 is 2D \(m, k\). x2 must be 2D \(k, n\). The axes meet the input parameter requirements of the mm operator and are equal, and the value range of the k axis is \[256, 65535\). bias does not support non-zero input.
- x1/x2 supports empty tensor scenarios. m and n can be empty, k cannot be empty, and the following conditions must be met:
    - m is empty, k is not empty, and n is not empty.
    - If m is not empty, k is not empty, and n is empty;
    - If m is empty, k is not empty, and n is empty.
- The data type of x1 and x2 must be the same as that of output.
- Matrix x2 can be transposed or not transposed, and matrix x1 can only be not transposed.
- The output is two-dimensional, and its shape is \(m*rank\_size, n\). rank\_size indicates the number of devices.
- Ascend 910B AI Processor: 2, 4, and 8 devices are supported, and only all-mesh networking of HCCS links is supported.
- Ascend 910_93 AI Processor: 2, 4, 8, and 16 devices are supported, and only double-ring networking of HCCS links is supported.
- AllGatherMatmul, MatmulReduceScatter, or MatmulAllReduce operators in a model support only the same communicator.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](common/compilation_running_sample.md).

```Cpp
#include <thread>
#include <iostream>
#include <vector>
#include "aclnnop/aclnn_all_gather_matmul.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int DEV_NUM = 8;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
  };

int launchOneThread_AllGatherMm(Args &args)
{
    int ret = aclrtSetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
    std::vector<int64_t> x1Shape = {128, 256};
    std::vector<int64_t> x2Shape = {256, 512};
    std::vector<int64_t> biasShape = {512};
    std::vector<int64_t> outShape = {128 * DEV_NUM, 512};
    std::vector<int64_t> gatherOutShape = {128 * DEV_NUM, 256};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *gatherOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *out = nullptr;
    aclTensor *gatherOut = nullptr;

    int64_t gatherIndex = 0;
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long outShapeSize = GetShapeSize(outShape);
    long long gatherOutShapeSize = GetShapeSize(gatherOutShape);

    std::vector<int16_t> x1HostData(x1ShapeSize, 0);
    std::vector<int16_t> x2HostData(x2ShapeSize, 0);
    std::vector<int16_t> biasHostData(biasShapeSize, 0);
    std::vector<int16_t> outHostData(outShapeSize, 0);
    std::vector<int16_t> gatherOutHostData(gatherOutShapeSize, 0);

    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gatherOutHostData, gatherOutShape, &gatherOutDeviceAddr,
        aclDataType::ACL_FLOAT16, &gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //Call the first-phase API.
    ret = aclnnAllGatherMatmulGetWorkspaceSize(
        x1, x2, bias, hcomName, gatherIndex, commTurn, streamMode, out, gatherOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAllGatherMatmulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    //Call the second-phase API.
    ret = aclnnAllGatherMatmul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmul failed. ret = %d \n", ret); return ret);
    // (Fixed expression) Synchronously wait until the task is complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);
    LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmul execute successfully.\n", args.rankId);
    // Release device resources. Modify the configuration based on the API definition.
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (gatherOut != nullptr) {
        aclDestroyTensor(gatherOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (gatherOutDeviceAddr != nullptr) {
        aclrtFree(gatherOutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    // Initialize the collective communicator.
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    // Start multiple threads.
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread_AllGatherMm, std::ref(args[rankId])));    
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    for (int i = 0; i < DEV_NUM; i++) {
        auto hcclRet = HcclCommDestroy(comms[i]);
        CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestory failed. ret = %d \n", ret); return -1);
    }
    aclFinalize();
    return 0;
}
```
