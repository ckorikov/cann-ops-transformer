声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaCompress

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。

- 计算公式：

    Nsa Compress正向计算公式如下：
$$
\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}
$$


## 实现原理

图1 训练计算流程图

![NsaCompress图](./fig/NsaCompress正向压缩示意图.png)

按照NSA Compress正向计算流程实现，整体计算流程如下：

1. 输入weight经broadcast， cast后常驻UB。输入Input与输入actSeqLenOptional经过tiling计算后，每次拷贝subInput到UB上，再经过cast成初步subInput。然后subInput与weight做vmul，后续在token维度做二分reduce得到中间计算结果overlap。由于每次搬运到UB上的subInput可能参与多个压缩后输出Token计算，因此每个subInput与weight不同偏移位置进行vmul/reduce后得到不同overlap，从而达到减少搬运subInput次数同时压缩多个输出Token效果

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnNsaCompressGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompress”接口执行计算。

* `aclnnStatus aclnnNsaCompressGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclIntArray *actSeqLenOptional, char *layoutOptional, int64_t compressBlockSize, int64_t compressStride, int64_t actSeqLenType, aclTensor *output, uint64_t *workspaceSize aclOpExecutor **executor)`
* `aclnnStatus aclnnNsaCompress(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNsaCompressGetWorkspaceSize

- **参数说明：**

  - input（aclTensor \*，计算输入）：Device侧的aclTensor，表示待压缩张量。shape支持[T, N, D]， 数据类型支持FLOAT16、BFLOAT16，数据类型与weight数据类型一致，[数据格式](common/数据格式.md)支持ND；支持[非连续的Tensor](common/非连续的Tensor.md)， 不支持空Tensor。综合约束请见[约束说明](#约束说明)。

    **说明：**
    input数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N；
    其中T是B和S合轴紧密排列的数据（每个batch的actSeqLen）、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。

  - weight（aclTensor \*，计算输入）：Device侧的aclTensor，表示压缩权重。shape支持[compressBlockSize, N]，weight与input的shape满足broadcast关系， 数据类型与input数据类型保持一致，[数据格式](common/数据格式.md)支持ND， 支持[非连续的Tensor](common/非连续的Tensor.md)， 不支持空Tensor。综合约束请见[约束说明](#约束说明)。

  - actSeqLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，可选参数，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND， 描述了每个Batch对应的S大小，当前不能为空；综合约束请见[约束说明](#约束说明)。

  - layoutOptional（char \*，计算输入）：Host侧的string，可选参数，代表输入input的数据排布格式，支持BSH、SBH、BSND、BNSD、TND。当前仅支持TND。

  - compressBlockSize（int64\_t，计算输入）：Host侧的int64_t，压缩滑窗大小。

  - compressStride（int64\_t，计算输入）：Host侧的int64_t，两次压缩滑窗间隔大小。

  - actSeqLenType（int64\_t，计算输入）：Host侧的int64_t，可取值0或1，0代表actSeqLenOptional中数值为前继batch的系列大小的cumsum结果（累积和），1代表actSeqLenOptional中数值为每个batch中序列大小，当前仅支持0。

  - output（aclTensor*，计算输出）： Device侧的aclTensor，压缩后的结果。shape支持[T, N, D]，数据类型与input保持一致，数据格式支持ND。支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor。

  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入input、weight、actSeqLenOptional或output是空指针
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：1. input和weight的数据类型不在支持的范围之内
                                          2. input和weight的shape无法做broadcast
                                          3. layoutOptional不合法
  ```

### aclnnNsaCompress

- **参数说明：**

  -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressGetWorkspaceSize获取。
  -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- input和weight需要满足broadcast关系，input.shape[1]=weight.shape[1]，不支持input、weight为空输入。
- actSeqLenType目前仅支持取值0，即actSeqLenOptional需要是前缀和模式。
- actSeqLenOptional目前不支持为空。
- layoutOptional目前仅支持TND，此时input.shape[0]必须等于actSeqLenOptional[-1]。
- input.shape[1]=weight.shape[1]，需要小于等于128。
- input.shape[2]必须是16的倍数，上限256。
- weight.shape[0]=compressBlockSize，必须是16的倍数，上限128。
- compressStride必须是16的整数倍，并且compressBlockSize>=compressStride。

## 算子原型

```c++
REG_OP(NsaCompress)
    .INPUT(input, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actSeqLenOptional, TensorType({DT_INT64, DT_INT64}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(layoutOptional, String)
    .ATTR(compressBlockSize, Int, 32)
    .ATTR(compressStride, Int, 16)
    .ATTR(actSeqLenType, Int)
    .OP_END_FACTORY_REG(NsaCompress)
```

参数解释请参见**算子执行接口**。

## 调用示例

该融合算子有两种调用方式：

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_nsa_compress](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

  ```c++

    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_nsa_compress.h"

    #define CHECK_RET(cond, return_expr) \
        do                               \
        {                                \
            if (!(cond))                 \
            {                            \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do                                  \
        {                                   \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shapeSize = 1;
        for (auto i : shape)
        {
            shapeSize *= i;
        }
        return shapeSize;
    }

    void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
    {
        auto size = GetShapeSize(shape);
        std::vector<aclFloat16> resultData(size, 0);
        auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
        for (int64_t i = 0; i < size; i++)
        {
            LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
        }
    }

    int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
    {
        // 固定写法，AscendCL初始化
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(context, deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetCurrentContext(*context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        // 调用aclrtMalloc申请device侧内存
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // 调用aclCreateTensor接口创建aclTensor
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                                shape.size(), *deviceAddr);
        return ACL_SUCCESS;
    }
    int main()
    {
        // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
        // 根据自己的实际device填写deviceId
        int32_t deviceId = 0;
        aclrtContext context;
        aclrtStream stream;
        auto ret = Init(deviceId, &context, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. 构造输入与输出，需要根据API的接口自定义构造
        void *inputDeviceAddr = nullptr;
        void *weightDeviceAddr = nullptr;
        void *outputDeviceAddr = nullptr;

        aclTensor *input = nullptr;
        aclTensor *weight = nullptr;
        aclIntArray *actSeqLenOptional = nullptr;
        aclTensor *output = nullptr;

        // 自定义输入与属性
        int64_t compressBlockSize = 32;
        int64_t compressStride = 32;
        int64_t actSeqLenType = 0; // 0是前缀和模式，1是count计数模式
        char *layout = "TND";
        int32_t batchSize = 1;
        int32_t sampleLen = 64;
        int32_t headNum = 4;
        int32_t headDim = 32;

        std::vector<int64_t> inputShape = {batchSize * sampleLen, headNum, headDim};
        std::vector<int64_t> weightShape = {compressBlockSize, headNum};
        std::vector<int64_t> actSeqShape = {batchSize};
        std::vector<aclFloat16> inputHostData(batchSize * sampleLen * headNum * headDim);
        std::vector<aclFloat16> weightHostData(compressBlockSize * headNum);
        std::vector<int64_t> actSeqHostData(batchSize);

        for (int i = 0; i < inputHostData.size(); i++)
        {
            inputHostData[i] = aclFloatToFloat16(1.0);
        }
        for (int i = 0; i < weightHostData.size(); i++)
        {
            weightHostData[i] = aclFloatToFloat16(1.0);
        }

        int outputNum = 0;
        int preActSeqLen = 0;
        for (int i = 0; i < batchSize; i++)
        {
            if (actSeqLenType == 0)
            {
                actSeqHostData[i] = sampleLen + preActSeqLen;
                preActSeqLen = actSeqHostData[i];
            }
            else if (actSeqLenType == 1)
            {
                actSeqHostData[i] = sampleLen;
            }
            if (sampleLen >= compressBlockSize)
            {
                outputNum += (sampleLen - compressBlockSize) / compressStride + 1;
            }
        }

        std::vector<int64_t> outputShape = {outputNum, headNum, headDim};
        std::vector<aclFloat16> outputHostData(outputNum * headNum * headDim);

        ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        actSeqLenOptional = aclCreateIntArray(actSeqHostData.data(), actSeqHostData.size());

        ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // 3. 调用CANN算子库API，需要修改为具体的Api名称
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        // 调用aclnnNsaCompressGetWorkspaceSize第一段接口
        ret = aclnnNsaCompressGetWorkspaceSize(input, weight, actSeqLenOptional, layout, compressBlockSize, compressStride,
                                            actSeqLenType, output, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        // 根据第一段接口计算出的workspaceSize申请device内存
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }

        // 调用aclnnNsaCompress第二段接口
        ret = aclnnNsaCompress(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompress failed. ERROR: %d\n", ret); return ret);

        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
        PrintOutResult(outputShape, &outputDeviceAddr);

        // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
        aclDestroyTensor(input);
        aclDestroyTensor(weight);
        aclDestroyIntArray(actSeqLenOptional);
        aclDestroyTensor(output);

        // 7. 释放device资源
        aclrtFree(inputDeviceAddr);
        aclrtFree(weightDeviceAddr);
        // aclrtFree(actSeqDeviceAddr);
        aclrtFree(outputDeviceAddr);
        if (workspaceSize > 0)
        {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }

  ```
