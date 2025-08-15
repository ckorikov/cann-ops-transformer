声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaSelectedAttention

## 支持的产品型号

- <term>Atlas A2 训练系列产品</term>

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。

- 计算公式：
  选择注意力的正向计算公式如下：

  $$
  selected\_key = Gather(key, topk\_indices[i]),0<=i<selected\_block\_count \\
  selected\_value = Gather(value, topk\_indices[i]),0<=i<selected\_block\_count
  $$
  
  $$
  attention\_out = Softmax(Mask(scale * (query @ selected\_key^T), atten\_mask)) @ selected\_value
  $$

## 实现原理

按照NativeSparseAttention的selected-attention正向计算流程实现，整体计算流程如下：

1. `key`和`value`根据`topk_indices`选择`selected_block_count`个长度为`selected_block_size`的块，分别得到`selected_key`和`selected_value`。
2. `query`与转置后的`selected_key`做matmul计算后得到最初步的`attention_score`，然后乘以缩放系数`scale_value`。此时的结果通过`atten_mask`进行select操作，将`atten_mask`中为true的位置进行遮蔽，得到结果`masked_attention_score`，即`atten_mask`中为true的位置在select后结果为负的极小值，经过softmax计算之后变成0从而达到遮蔽效果，softmax计算后得到`softmax_res`和两个输出`softmax_max`、`softmax_sum`。
3. 最后`softmax_res`与`selected_value`做matmul计算得到最终的结果`attention_out`。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnNsaSelectedAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaSelectedAttention”接口执行计算。

- `aclnnStatus aclnnNsaSelectedAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *topkIndices, const aclTensor *attenMaskOptional,const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue, int64_t headNum, char *inputLayout, int64_t sparseMode, int64_t selectedBlockSize, int64_t selectedBlockCount, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnNsaSelectedAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

### aclnnNsaSelectedAttentionGetWorkspaceSize

- **参数说明：**
  
  - query（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的query，数据类型支持BFLOAT16、FLOAT16，数据类型与`key/value`的数据类型一致，[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
  - key（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的key，数据类型支持BFLOAT16、FLOAT16，数据类型与`query/value`的数据类型一致，[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
  - value（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的value，数据类型支持BFLOAT16、FLOAT16，数据类型与`query/key`的数据类型一致，[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
  - topkIndices（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的topk\_indices，shape需为[T_q, N_kv, selected_block_count], 表示所选数据的索引，数据类型支持INT32，[数据格式](common/数据格式.md)支持ND，综合约束请见[约束说明](#约束说明)。
  - attenMaskOptional（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的atten\_mask，数据类型支持BOOL、UINT8取值为`true/1`代表该位不参与计算（不生效），为`false/0`代表该位参与计算，[数据格式](common/数据格式.md)支持ND，输入shape类型需为[S_q, S_kv]；综合约束请见[约束说明](#约束说明)。
  - actualSeqQLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，长度等于batchsize。该数组表示`query`每个Batch S的累加和长度，假设输入真实的S长度分别为\[2,2,3,2\]，则传入的actualSeqQLenOptional为\[2,4,7,9\]。在TND排布时需要输入，其余场景下输入nullptr。
  - actualSeqKvLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，长度等于batchsize。该数组表示`key/value`每个Batch S的累加和长度，假设输入真实的S长度分别为\[1024,1024,1024,1024\]，则传入的actualSeqKvLenOptional为\[1024,2048,3072,4096\]。在TND排布时需要输入，其余场景下输入nullptr。
  - scaleValue（double，计算输入）：Host侧的double，公式中的scale，代表缩放系数，数据类型支持DOUBLE，一般设置为`D^-0.5`，其中D为输入`query`的head维度。
  - headNum（int64\_t，计算输入）：Host侧的int64_t，代表head个数，即输入`query`的N轴长度，数据类型支持INT64；综合约束请见[约束说明](#约束说明)。
  - inputLayout（string \*，计算输入）：Host侧的string，数据类型支持String，代表输入`query`、`key`、`value`的数据排布格式，当前仅支持TND，其中T表示各batch S的长度累加和，N表示Head-Num，D表示Head-Dim。
  - selectedBlockSize（int64\_t，计算输入）：Host侧的int64_t，表示select的每个block长度。
  - selectedBlockCount（int64\_t，计算输入）：Host侧的int64_t，公式中的selected\_block\_count，表示select block的数量。
  - sparseMode（int64\_t，计算输入）：Host侧的int64_t，表示sparse的模式。数据类型支持INT32。目前支持sparseMode=0或者2。sparse不同模式的详细说明请参见[sparse模式说明](./common/sparse_mode参数说明.md)。
  - softmaxMaxOut（aclTensor \*，计算输出）：Device侧的aclTensor，Softmax计算的Max中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[T_q, N_q, 8]，[数据格式](common/数据格式.md)支持ND。
  - softmaxSumOut（aclTensor \*，计算输出）：Device侧的aclTensor，Softmax计算的Sum中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[T_q, N_q, 8]，[数据格式](common/数据格式.md)支持ND。
  - attentionOut（aclTensor \*，计算输出）：Device侧的aclTensor，计算公式的最终输出。数据类型支持BFLOAT16、FLOAT16，输出数据类型与`query`保持一致, shape类型为[T_q, N_q, D_v]，[数据格式](common/数据格式.md)支持ND。
  - workspaceSize（uint64\_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：1. query、key、value、attenMaskOptional、softmaxMaxOut、softmaxSumOut、attentionOut的数据类型和数据格式不在支持的范围内。
                                          2. input_layout输入的类型不在支持的范围内。
  ```

### aclnnNsaSelectedAttention

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNsaSelectedAttentionGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value的batchsize必须相等，即要求传入的actualSeqQLenOptional和actualSeqKvLenOptional具有相同的长度。
- 输入query、key、value的D：Head-Dim必须满足（D_q == D_k && D_k >= D_v）。
- 输入query、key、value的数据类型必须一致。
- 输入query、key、value的input_layout必须一致。
- sparseMode目前支持0和2。
- selectedBlockSize支持<=128且满足16的整数倍。
- selectedBlockCount支持<=32。
- inputLayout目前仅支持TND。
- 支持输入query的N和key/value的N不相等，但必须成比例关系，即N_q / N_kv必须是非0整数，称为G（group），且需满足G <= 32。
- 当attenMaskOptional输入为nullptr时，sparseMode参数不生效，固定为全计算。
- 关于数据shape的约束，以inputLayout的TND举例（注：T等于各batch S的长度累加和。当各batch的S相等时，T=B*S）。其中：
  
  - B（Batchsize）：取值范围为1\~1024。
  - N（Head-Num）：取值范围为1\~128。
  - G（Group）：取值范围为1\~32。
  - S（Seq-Length）：取值范围为1\~128K。同时需要满足S_kv >= selectedBlockSize * selectedBlockCount，且S_kv长度为selectedBlockSize的整数倍。
  - D（Head-Dim）：D_qk=192，D_v=128。

## 算子原型

```c++
REG_OP(NsaSelectedAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(topk_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({ DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({ DT_INT64}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(scale_value, Float, 1)
    .ATTR(head_num, Int, 1)
    .ATTR(sparse_mode, Int, 0)
    .REQUIRED_ATTR(input_layout, String)
    .REQUIRED_ATTR(selected_block_size, Int)
    .REQUIRED_ATTR(selected_block_count, Int)
    .OP_END_FACTORY_REG(NsaSelectedAttention)
```

参数解释请参见**算子执行接口**。

## 调用示例

通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_selected_attention.h"


#define CHECK_RET(cond, return_expr)                   \
    do {                                               \
        if (!(cond)) {                                 \
            return_expr;                               \
        }                                              \
    } while (0)

#define LOG_PRINT(message, ...)                        \
    do {                                               \
        printf(message, ##__VA_ARGS__);                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T> void CopyOutResult(int64_t outIndex, std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    if(outIndex == 2) {
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("attention out result is: %f\n", i, resultData[i]);
        }
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
                                  aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
                                  aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
                                  aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
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

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FreeResource(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *attentionOut, aclTensor *softmaxMax,
    aclTensor *softmaxSum, void *qDeviceAddr, void *kDeviceAddr, void *vDeviceAddr, void *attentionOutDeviceAddr,
    void *softmaxMaxDeviceAddr, void *softmaxSumDeviceAddr, uint64_t workspaceSize, void *workspaceAddr,
    int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (v != nullptr) {
        aclDestroyTensor(v);
    }
    if (attentionOut != nullptr) {
        aclDestroyTensor(attentionOut);
    }
    if (softmaxMax != nullptr) {
        aclDestroyTensor(softmaxMax);
    }
    if (softmaxSum != nullptr) {
        aclDestroyTensor(softmaxSum);
    }

    // 释放device资源
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (vDeviceAddr != nullptr) {
        aclrtFree(vDeviceAddr);
    }
    if (attentionOutDeviceAddr != nullptr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (softmaxMaxDeviceAddr != nullptr) {
        aclrtFree(softmaxMaxDeviceAddr);
    }
    if (softmaxSumDeviceAddr != nullptr) {
        aclrtFree(softmaxSumDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
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
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_nsa_selected_attention 分支下生成
    // query、key、value对应的shape值，并重新gen data，再执行
    int64_t batch = 2;
    int64_t s1 = 512;
    int64_t s2 = 2048;
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 4;
    int64_t n2 = 4;
    std::vector<int64_t> qShape = {batch * s1, n2 * g, d1};
    std::vector<int64_t> kShape = {batch * s2, n2, d1};
    std::vector<int64_t> vShape = {batch * s2, n2, d2};
    std::vector<int64_t> topKIndicesShape = {batch * s1, n2, 16};
    std::vector<int64_t> attentionOutShape = {batch * s1, n2 * g, d2};
    std::vector<int64_t> softmaxMaxShape = {batch * s1, n2 * g, 8};
    std::vector<int64_t> softmaxSumShape = {batch * s1, n2 * g, 8};
    
    double scaleValue = 1.0;
    int64_t headNum = 16;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    int64_t sparseMod = 2;
    char layOut[] = "TND";

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *topKIndicesDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *softmaxMaxDeviceAddr = nullptr;
    void *softmaxSumDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *topKIndices = nullptr;
    aclTensor *attenMaskOptional = nullptr;
    aclTensor *softmaxMax = nullptr;
    aclTensor *softmaxSum = nullptr;
    aclTensor *attentionOut = nullptr;

    std::vector<int64_t> actualSeqQLenVec = {512, 1024};
    std::vector<int64_t> actualSeqKvLenVec = {2048, 4096};
    aclIntArray *actualSeqQLenOptional = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
    aclIntArray *actualSeqKvLenOptional = aclCreateIntArray(actualSeqKvLenVec.data(), actualSeqKvLenVec.size());

    std::vector<aclFloat16> qHostData(GetShapeSize(qShape), 1);
    std::vector<aclFloat16> kHostData(GetShapeSize(kShape), 1);
    std::vector<aclFloat16> vHostData(GetShapeSize(vShape), 1);
    std::vector<int32_t> topkIndicesHostData(GetShapeSize(topKIndicesShape), 2);
    std::vector<float> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0);
    std::vector<float> softmaxMaxHostData(GetShapeSize(softmaxMaxShape), 0.0);
    std::vector<float> softmaxSumHostData(GetShapeSize(softmaxSumShape), 0.0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    // 创建acl Tensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(topkIndicesHostData, topKIndicesShape, &topKIndicesDeviceAddr, aclDataType::ACL_INT32, &topKIndices);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnNsaSelectedAttention第一段接口
    ret = aclnnNsaSelectedAttentionGetWorkspaceSize(
        q, k, v, topKIndices, attenMaskOptional, actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, headNum,
        layOut, sparseMod, selectedBlockSize, selectedBlockCount, softmaxMax, softmaxSum, attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                deviceId, &context, &stream);
            return ret);
    }

    // 调用aclnnNsaSelectedAttention第二段接口
    ret = aclnnNsaSelectedAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttention failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    CopyOutResult<float>(0, softmaxMaxShape, &softmaxMaxDeviceAddr);
    CopyOutResult<float>(1, softmaxSumShape, &softmaxSumDeviceAddr);
    CopyOutResult<aclFloat16>(2, attentionOutShape, &attentionOutDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
        attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
        deviceId, &context, &stream);

    return 0;
}
```