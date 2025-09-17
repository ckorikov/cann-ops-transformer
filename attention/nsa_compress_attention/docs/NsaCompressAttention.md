声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaCompressAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能说明

-   **算子功能**：NSA中compress attention以及select topk索引计算。论文：https://arxiv.org/pdf/2502.11089

-   **计算公式**：压缩block大小：$l$，select block大小：$l'$，压缩stride大小：$d$

$$
P_{cmp} = Softmax(query*key^T) \\
$$

$$
attentionOut = Softmax(atten\_mask(scale*query*key^T, atten\_mask)))*value
$$

$$
P_{slc}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l/d-1}P_{cmp} [l'/d*j-m-n],
$$

$$
P_{slc'} = \sum_{h=1}^{H}P_{slc}^{h}
$$

$$
P_{slc'} = topk\_mask(P_{slc'})
$$

$$
topkIndices = topk(P_{slc'})
$$

NsaCompressAttention输入query、key、value的数据排布格式支持从多种维度排布解读，可通过inputLayout传入，当前仅支持TND。
- B：表示输入样本批量大小（Batch）
- T：B和S合轴紧密排列的长度
- S：表示输入样本序列长度（Seq-Length）
- H：表示隐藏层的大小（Head-Size）
- N：表示多头数（Head-Num）
- D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）


## 算子执行接口

* `aclnnStatus aclnnNsaCompressAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *attenMaskOptional, const aclTensor *topkMaskOptional, const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualCmpSeqKvLenOptional, const aclIntArray *actualSelSeqKvLenOptional, double scaleValue, int64_t headNum, char *inputLayout, int64_t sparseMode, int64_t compressBlockSize, int64_t compressStride, int64_t selectBlockSize, int64_t selectBlockCount, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *attentionOutOut, const aclTensor *topkIndicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnNsaCompressAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNsaCompressAttentionGetWorkspaceSize

- **参数说明：**
    - query（aclTensor *，计算输入）：Device侧的aclTensor，公式中的query，数据类型支持FLOAT16、BFLOAT16。[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
    - key（aclTensor *，计算输入）：Device侧的aclTensor，公式中的key，数据类型支持FLOAT16、BFLOAT16。[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
    - value（aclTensor *，计算输入）：Device侧的aclTensor，公式中的value，数据类型支持FLOAT16、BFLOAT16。[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
    - attenMaskOptional（aclTensor *，计算输入）：Device侧的aclTensor，公式中的atten_mask，数据类型支持BOOL，[数据格式](common/数据格式.md)支持ND，输入shape需为[S,S]，TND场景只支持SS格式，SS分别是max(Sq)和max(CmqSkv)；综合约束请见[约束说明](#约束说明)。
    - actualSeqQLenOptional（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，描述了每个Batch对应的query S大小(Sq)；综合约束请见[约束说明](#约束说明)。
    - actualCmpSeqKvLenOptional（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，描述了compress attention的每个Batch对应的key/value S大小(CmpSkv)；综合约束请见[约束说明](#约束说明)。
    - actualSelSeqKvLenOptional（aclIntArray*，计算输入）：Host侧的aclIntArray，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，描述了经importance score计算压缩后的每个Batch对应的key/value S大小(SelSkv)，对应公式中的P_slc'第0维；综合约束请见[约束说明](#约束说明)。
    - topkMaskOptional（aclTensor *，计算输入）：Device侧的aclTensor，公式中的topk_mask，数据类型支持BOOL，[数据格式](common/数据格式.md)支持ND，输入shape类型需为[S,S]，TND场景只支持SS格式，SS分别是max(Sq)和max(SelSkv)；综合约束请见[约束说明](#约束说明)。如不使用该参数可传入nullptr。
    - scaleValue（double，计算输入）：Host侧的double。公式中的scale，代表缩放系数，数据类型支持DOUBLE，一般设置为D^-0.5。
    - headNum（int64_t，计算输入）：Host侧的int64_t，数据类型支持INT64，代表query的head个数。
    - inputLayout（string*，计算输入）：Host侧的string，数据类型支持String，代表输入query、key、value的数据排布格式，当前支持TND。
    - sparseMode（int64_t，计算输入）：Host侧的int64_t。数据类型支持INT64。当前仅支持0和1；sparse不同模式的详细说明请参见[sparse模式说明](./common/sparse_mode参数说明.md)。
    - compressBlockSize（int64\_t，计算输入）：Host侧的int64_t，压缩滑窗大小，对应公式中的l。
    - compressStride（int64\_t，计算输入）：Host侧的int64_t，两次压缩滑窗间隔大小，对应公式中的d。
    - selectBlockSize（int64\_t，计算输入）：Host侧的int64_t，选择块大小，对应公式中的l'。
    - selectBlockCount（int64\_t，计算输入）：Host侧的int64_t，选择块个数，对应公式中topK选择个数。
    - softmaxMaxOut（aclTensor\*，计算输出）：Device侧的aclTensor，Softmax计算的Max中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[T,N,8]。[数据格式](common/数据格式.md)支持ND。
    - softmaxSumOut（aclTensor\*，计算输出）：Device侧的aclTensor，Softmax计算的Max中间结果，用于反向计算。数据类型支持FLOAT，输出的shape类型为[T,N,8]。[数据格式](common/数据格式.md)支持ND。
    - attentionOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的attentionOut，数据类型支持FLOAT16、BFLOAT16，数据类型和shape类型与query保持一致，[数据格式](common/数据格式.md)支持ND。
    - topkIndicesOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的topkIndices，数据类型支持INT32，输出的shape类型为[T,N2,selectBlockCount]。
    - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 输入query，key，value 传入的是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. query，key，value 数据类型不在支持的范围之内。
                                        2. inputLayout不合法。
                                        3. sparseMode不合法
  ```


### aclnnNsaCompressAttention

- **参数说明：**

  -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressAttentionGetWorkspaceSize获取。
  -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- compressBlockSize、compressStride、selectBlockSize必须是16的整数倍，并且满足：compressBlockSize>=compressStride && selectBlockSize>=compressBlockSize && selectBlockSize%compressStride==0
- compressBlockSize：16对齐，支持到128
- compressStride：16对齐，支持到64
- selectBlockSize：16对齐，支持到128
- selectBlockCount：支持[1~32] && selectBlockCount <= min(SelSkv)
- actualSeqQLenOptional, actualCmpSeqKvLenOptional, actualSelSeqKvLenOptional需要是前缀和模式；且TND格式下必须传入。
- 由于UB限制，CmpSkv需要满足以下约束：CmpSkv <= 14000
- SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)
- layoutOptional目前仅支持TND。
- 输入query、key、value的数据类型必须一致。
- 输入query、key、value的batchSize必须相等。
- 输入query、key、value的headDim必须满足：qD == kD && kD >= vD
- 输入query、key、value的inputLayout必须一致。
- 输入query的headNum为N1，输入key和value的headNum为N2，则N1 >= N2 && N1 % N2 == 0
- 设G = N1 / N2，G需要满足以下约束：G < 128 && 128 % G == 0
- attenMask和topkMask的使用需符合论文描述

## 算子原型

```c++
REG_OP(NsaCompressAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_cmp_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_sel_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(topk_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(topk_indices_out, TensorType({DT_INT32}))
    .ATTR(scale_value, Float, 1.0)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(sparse_mode, Int, 0)
    .REQUIRED_ATTR(compress_block_size, Int)
    .REQUIRED_ATTR(compress_stride, Int)
    .REQUIRED_ATTR(select_block_size, Int)
    .REQUIRED_ATTR(select_block_count, Int)
    .OP_END_FACTORY_REG(NsaCompressAttention)
```
参数解释请参见**算子执行接口**。

## 调用示例

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_nsa_compress_attention](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

    ```c++
    #include <iostream>
    #include <vector>
    #include <cstring>
    #include "acl/acl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_nsa_compress_attention.h"

    using namespace std;

    #define CHECK_RET(cond, return_expr) \
      do {                               \
        if (!(cond)) {                   \
        return_expr;                   \
        }                                \
      } while (0)

    #define LOG_PRINT(message, ...)     \
      do {                              \
        printf(message, ##__VA_ARGS__); \
      } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
        shapeSize *= i;
      }
      return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
      // 固定写法，AscendCL初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
    }

    void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
      auto size = GetShapeSize(shape);
      std::vector<float> resultData(size, 0);
      auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                            *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
      for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
      }
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      
      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                    shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

    int main() {
      // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t T1 = 1024;
      int64_t T2 = 64;
      int64_t N1 = 16;
      int64_t N2 = 4;
      int64_t D1 = 192;
      int64_t D2 = 128;
      int64_t selectBlockSize = 64;
      int64_t selectBlockCount = 16;
      int64_t compressBlockSize = 32;
      int64_t compressStride = 16;
      std::vector<int64_t> qShape = {T1, N1, D1};
      std::vector<int64_t> kShape = {T2, N2, D1};
      std::vector<int64_t> vShape = {T2, N2, D2};
      std::vector<int64_t> attenmaskShape = {T1, T2};                        //[maxS1, maxS2]
      std::vector<int64_t> topkmaskShape = {T1, T1 / selectBlockSize};       //[maxS1, maxSelS2]
      std::vector<int64_t> softmaxMaxShape = {T1, N1, 8};
      std::vector<int64_t> softmaxSumShape = {T1, N1, 8};
      std::vector<int64_t> attenOutShape = {T1, N1, D2};                     //[T1, N1, D2]
      std::vector<int64_t> topkIndicesOutShape = {T1, N2, selectBlockCount}; //[T1, N2, selectBlockCount]

      void* qDeviceAddr = nullptr;
      void* kDeviceAddr = nullptr;
      void* vDeviceAddr = nullptr;
      void* attenmaskDeviceAddr = nullptr;
      void* topkmaskDeviceAddr = nullptr;
      void* softmaxMaxDeviceAddr = nullptr;
      void* softmaxSumDeviceAddr = nullptr;
      void* attentionOutDeviceAddr = nullptr;
      void* topkIndicesOutDeviceAddr = nullptr;

      aclTensor* q = nullptr;
      aclTensor* k = nullptr;
      aclTensor* v = nullptr;
      aclTensor* attenmask = nullptr;
      aclTensor* topkmask = nullptr;
      aclTensor* softmaxMax = nullptr;
      aclTensor* softmaxSum = nullptr;
      aclTensor* attentionOut = nullptr;
      aclTensor* topkIndicesOut = nullptr;

      std::vector<op::fp16_t> qHostData(T1 * N1 * D1, 1.0);
      std::vector<op::fp16_t> kHostData(T2 * N2 * D1, 1.0);
      std::vector<op::fp16_t> vHostData(T2 * N2 * D2, 1.0);
      std::vector<uint8_t> attenmaskHostData(T1 * T2, 0);
      std::vector<uint8_t> topkmaskHostData(T1 * (T1 / selectBlockSize), 0);
      std::vector<float> softmaxMaxHostData(N1 * T1 * 8, 1.0);
      std::vector<float> softmaxSumHostData(N1 * T1 * 8, 1.0);
      std::vector<op::fp16_t> attenOutHostData(T1 * N1 * D2, 1.0);
      std::vector<int32_t> topkIndicesHostData(T1 * N2 * selectBlockCount, 1);

      ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(topkmaskHostData, topkmaskShape, &topkmaskDeviceAddr, aclDataType::ACL_UINT8, &topkmask);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(attenOutHostData, attenOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(topkIndicesHostData, topkIndicesOutShape, &topkIndicesOutDeviceAddr, aclDataType::ACL_INT32, &topkIndicesOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      std::vector<int64_t> actualSeqQLenVec(1, T1);
      auto actualSeqQLen = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
      std::vector<int64_t> actualCmpKvSeqVec(1, T2);
      auto actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqVec.data(), actualCmpKvSeqVec.size());
      std::vector<int64_t> actualSelKvSeqVec(1, T1 / selectBlockSize);
      auto actualSelKvSeqLen = aclCreateIntArray(actualSelKvSeqVec.data(), actualSelKvSeqVec.size());

      double scale = 1.0;
      int64_t headNum = N1;
      char inputLayout[5] = {'T', 'N', 'D', 0};
      int64_t sparseMode = 1;

      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 调用第一段接口
      ret = aclnnNsaCompressAttentionGetWorkspaceSize(q, k, v, attenmask, topkmask, actualSeqQLen, actualCmpKvSeqLen,
            actualSelKvSeqLen, scale, headNum, inputLayout, sparseMode, compressBlockSize, compressStride, selectBlockSize, selectBlockCount, 
            softmaxMax, softmaxSum, attentionOut, topkIndicesOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // 调用第二段接口
      ret = aclnnNsaCompressAttention(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttention failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      PrintOutResult(attenOutShape, &attentionOutDeviceAddr);
      PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
      PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
      PrintOutResult(topkIndicesOutShape, &topkIndicesOutDeviceAddr);

      // 6. 释放资源
      aclDestroyTensor(q);
      aclDestroyTensor(k);
      aclDestroyTensor(v);
      aclDestroyTensor(attenmask);
      aclDestroyTensor(topkmask);
      aclDestroyTensor(softmaxMax);
      aclDestroyTensor(softmaxSum);
      aclDestroyTensor(attentionOut);
      aclDestroyTensor(topkIndicesOut);
      aclrtFree(qDeviceAddr);
      aclrtFree(kDeviceAddr);
      aclrtFree(vDeviceAddr);
      aclrtFree(attenmaskDeviceAddr);
      aclrtFree(topkmaskDeviceAddr);
      aclrtFree(softmaxMaxDeviceAddr);
      aclrtFree(softmaxSumDeviceAddr);
      aclrtFree(attentionOutDeviceAddr);
      aclrtFree(topkIndicesOutDeviceAddr);
      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```