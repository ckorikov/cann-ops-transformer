声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaSelectedAttentionGrad

## 支持的产品型号

<term>Atlas A2 训练系列产品</term>

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。

- 计算公式：

    根据传入的topkIndice对keyIn和value选取数量为selectedBlockCount个大小为selectedBlockSize的数据重排，公式如下：

    $$
    selectedKey = Gather(key, topkIndices[i]),0<=i<selectedBlockCount \\
    selectedValue = Gather(value, topkIndices[i]),0<=i<selectedBlockCount
    $$

    接着，进行注意力机制的反向计算，计算公式为：

    $$
    V=P^TdY
    $$

    $$
    Q=\frac{((dS)*K)}{\sqrt{d}}
    $$

    $$
    K=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$


## 实现原理

整体计算流程如下：

1. 根据topkIndices挑选topK个KV，并将结果Gather后输出WorkSpace。同时需要保存attenMask相关变量。
2. 进入常规的FA计算流过程。
3. 直接输出dq的结果，dk和dv需要将FAG的计算结果输出根据indices放回原来的位置。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnNsaSelectedAttentionGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaSelectedAttentionGrad”接口执行计算。

* `aclnnStatus aclnnNsaSelectedAttentionGradGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *attentionOut, const aclTensor *attentionOutGrad, const aclTensor *softmaxMax, const aclTensor *softmaxSum, const aclTensor *topkIndices, const aclIntArray *actualSeqQLenOptional,const aclIntArray *actualSeqKvLenOptional, const aclTensor *attenMaskOptional, double scaleValue, int64_t selectedBlockSize, int64_t  selectedBlockCount, int64_t headNum, char *inputLayout, int64_t sparseMode, aclTensor *dqOut, aclTensor *dkOut, aclTensor *dvOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnNsaSelectedAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNsaSelectedAttentionGradGetWorkspaceSize

- **参数说明：**

  - query（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的输入Q，数据类型支持BFLOAT16和FLOAT16，shape支持3维，[数据格式](common/数据格式.md)支持ND，综合约束请见[约束说明](#约束说明)。
  - key（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的输入K，数据类型支持BFLOAT16和FLOAT16，数据类型与`query`一致，shape支持3维，[数据格式](common/数据格式.md)支持ND，综合约束请见[约束说明](#约束说明)。
  - value（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的输入V，数据类型支持BFLOAT16和FLOAT16，数据类型与`query`一致，shape支持3维，[数据格式](common/数据格式.md)支持ND，综合约束请见[约束说明](#约束说明)。
  - attentionOut（aclTensor \*，计算输入）：Device侧的aclTensor，注意力正向计算的最终输出，数据类型支持BFLOAT16和FLOAT16，数据类型与`query`一致，shape支持3维，[数据格式](common/数据格式.md)支持ND。
  - attentionOutGrad（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的输入dY，数据类型支持BFLOAT16和FLOAT16，数据类型与`query`一致，shape支持3维，[数据格式](common/数据格式.md)支持ND，综合约束请见[约束说明](#约束说明)。
  - softmaxMax（aclTensor \*，计算输入）：Device侧的aclTensor，注意力正向计算的中间输出，数据类型支持FLOAT，shape支持3维，[数据格式](common/数据格式.md)支持ND。综合约束请见[约束说明](#约束说明)。
  - softmaxSum（aclTensor \*，计算输入）：Device侧的aclTensor，注意力正向计算的中间输出，数据类型支持FLOAT，shape支持3维，[数据格式](common/数据格式.md)支持ND；综合约束请见[约束说明](#约束说明)。
  - topkIndices（aclTensor \*，计算输入）：Device侧的aclTensor，公式中的所选择KV的索引，数据类型支持INT32，shape支持3维，[数据格式](common/数据格式.md)支持ND，综合约束请见[约束说明](#约束说明)。
  - actualSeqQLenOptional（aclIntArray \*，计算输入）：数据类型支持INT64，描述了每个Batch对应的`query` S大小；综合约束请见[约束说明](#约束说明)。
  - actualSeqKvLenOptional（aclIntArray \*，计算输入）：数据类型支持INT64，描述了每个Batch对应的`key/value` S大小。
  - attenMaskOptional（aclTensor \*，计算输入）：Device侧的aclTensor，0代表保留，1代表掩掉的mask矩阵，**预留参数，暂不支持**。
  - scaleValue（double，计算输入）：公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。一般设置为$D^{-0.5}$。
  - selectedBlockSize（int64_t，计算输入）：所选择的块的大小，数据类型支持INT64，目前支持16~128，需对齐16。
  - selectedBlockCount（int64_t，计算输入）：所选择的块的数量，数据类型支持INT64，目前支持1~32。
  - headNum（int64_t，计算输入）：Host侧的int64_t，代表输入`query`的N轴长度，数据类型支持INT64；综合约束请见[约束说明](#约束说明)。
  - inputLayout（string \*，计算输入）：Host侧的string，代表输入`query`、`keyIn`、`value`的数据排布格式，支持TND。
    - `query`、`keyIn`、`value`数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、G（Group）表示组数，仅对Q生效，D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
  - sparseMode（int64_t，计算输入）：Host侧的int，表示sparse的模式。数据类型支持INT32。目前仅支持sparseMode=0或2。sparse不同模式的详细说明请参见[sparse模式说明](./common/sparse_mode参数说明.md)。
  - dqOut（aclTensor \*，计算输出）：Device侧的aclTensor，公式中的dQ，表示query的梯度，计算输出，数据类型支持BFLOAT16和FLOAT16，数据类型与query一致，shape支持3维，shape必须与query的shape一致,[数据格式](common/数据格式.md)支持ND。
  - dkOut（aclTensor \*，计算输出）：Device侧的aclTensor，公式中的dK，表示key的梯度，计算输出，数据类型支持BFLOAT16和FLOAT16，数据类型与query一致，shape支持3维，shape必须与key的shape一致，[数据格式](common/数据格式.md)支持ND。
  - dvOut（aclTensor \*，计算输出）：Device侧的aclTensor，公式中的dV，表示value的梯度，计算输出，数据类型支持BFLOAT16和FLOAT16，数据类型与query一致，shape支持3维，shape必须与value的shape一致，[数据格式](common/数据格式.md)支持ND。
  - workspaceSize（uint64\_t \*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入参数是必选输入，输出或者必选属性，且是空指针。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、dqOut、dkOut、dvOut的数据类型和数据格式不在支持的范围内。
  ```

### aclnnNsaSelectedAttentionGrad

- **参数说明：**

  -   workspace（void \*，入参）：在Device侧申请的workspace内存地址。
  -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNsaSelectedAttentionGradGetWorkspaceSize获取。
  -   executor（aclOpExecutor \*，入参）：op执行器，包含了算子计算流程。
  -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 该接口与pytorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value、attentionOut、attentionOutGrad的B（batchsize）必须相等。
- 输入key、value的N（numHead）必须一致。
- 输入query、attentionOut、attentionOutGrad的N（numHead）必须一致。
- 输入value、attentionOut、attentionOutGrad的D（HeadDim）必须一致。
- 输入query、key、value、attentionOut、attentionOutGrad的inputLayout必须一致。
- 关于数据shape的约束，以inputLayout的TND举例。其中：
  - T1：取值范围为1\~2M。T1表示query所有batch下S的和。
  - T2：取值范围为1\~2M。T2表示key、value所有batch下S的和。
  - B：取值范围为1\~2M。
  - N1：取值范围为1\~128。表示query的headNum。N1必须为N2的整数倍。
  - N2：取值范围为1\~128。表示key、value的headNum。
  - G：取值范围为1\~32。G = N1 / N2
  - S：取值范围为1\~128K。对于key、value的S 必须大于等于selectedBlockSize * selectedBlockCount, 且必须为selectedBlockSize的整数倍。
  - D：取值范围为192或128，支持K和V的D（HeadDim）不相等。
- 关于softmaxMax与softmaxSum参数shape的约束：\[T1, N1, 8\]。
- 关于topkIndices参数shape的约束：[T1, N2, selectedBlockCount]。


参数解释请参见**算子执行接口**。

## 调用示例

该融合算子有两种调用方式：

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_nsa_select_attention_grad](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_nsa_selected_attention_grad.h"
  
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
  
template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请Device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  
    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  
    // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
  
    // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }
  
int main() {
      // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
    // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t b = 1;
      int64_t s1 = 1;
      int64_t s2 = 1024;
      int64_t t1 = b * s1;
      int64_t t2 = b * s2;
      int64_t n1 = 1;
      int64_t n2 = 1;
      int64_t d = 192;
  
    int64_t sparseMode = 0;
      char inputLayout[5] = {'T', 'N', 'D', 0};
      double scaleValue = 1.0f;
      int64_t selectedBlockSize = 64;
      int64_t selectedBlockCount = 16;
      int32_t headNum = n1;
  
    std::vector<int64_t> queryShape = {t1, n1, d};
      std::vector<int64_t> keyShape = {t2, n2, d};
      std::vector<int64_t> valueShape = {t2, n2, d};
      std::vector<int64_t> attentionOutShape = {t1, n1, d};
      std::vector<int64_t> attentionOutGradShape = {t1, n1, d};
      std::vector<int64_t> softmaxMaxShape = {t1, n1, 8};
      std::vector<int64_t> softmaxSumShape = {t1, n1, 8};
      std::vector<int64_t> topkIndicesShape = {t1, n2, selectedBlockCount};
      std::vector<int64_t> actualSeqQLenOptionalShape = {b};
      std::vector<int64_t> actualSeqKvLenOptionalShape = {b};
      std::vector<int64_t> dqOutShape = {t1, n1, d};
      std::vector<int64_t> dkOutShape = {t2, n2, d};
      std::vector<int64_t> dvOutShape = {t2, n2, d};
  
    void* queryDeviceAddr = nullptr;
      void* keyDeviceAddr = nullptr;
      void* valueDeviceAddr = nullptr;
      void* attentionOutDeviceAddr = nullptr;
      void* attentionOutGradDeviceAddr = nullptr;
      void* softmaxMaxDeviceAddr = nullptr;
      void* softmaxSumDeviceAddr = nullptr;
      void* topkIndicesDeviceAddr = nullptr;
      void* dqOutDeviceAddr = nullptr;
      void* dkOutDeviceAddr = nullptr;
      void* dvOutDeviceAddr = nullptr;
  
    aclTensor* query = nullptr;
      aclTensor* key = nullptr;
      aclTensor* value = nullptr;
      aclTensor* attentionOut = nullptr;
      aclTensor* attentionOutGrad = nullptr;
      aclTensor* softmaxMax = nullptr;
      aclTensor* softmaxSum = nullptr;
      aclTensor* topkIndices = nullptr;
      aclTensor* dqOut = nullptr;
      aclTensor* dkOut = nullptr;
      aclTensor* dvOut = nullptr;
  
    std::vector<aclFloat16> queryHostData(GetShapeSize(queryShape), 2);
      std::vector<aclFloat16> keyHostData(GetShapeSize(keyShape), 2);
      std::vector<aclFloat16> valueHostData(GetShapeSize(valueShape), 2);
      std::vector<aclFloat16> attentionOutHostData(GetShapeSize(attentionOutShape), 2);
      std::vector<aclFloat16> attentionOutGradHostData(GetShapeSize(attentionOutGradShape), 2);
      std::vector<float> softmaxMaxHostData(GetShapeSize(softmaxMaxShape), 2);
      std::vector<float> softmaxSumHostData(GetShapeSize(softmaxSumShape), 2);
      std::vector<int32_t> topkIndicesHostData(GetShapeSize(topkIndicesShape), 1);
      std::vector<aclFloat16> dqOutHostData(GetShapeSize(dqOutShape), 2);
      std::vector<aclFloat16> dkOutHostData(GetShapeSize(dkOutShape), 2);
      std::vector<aclFloat16> dvOutHostData(GetShapeSize(dvOutShape), 2);
  
    for (int32_t i = 0; i < topkIndicesHostData.size(); i++) {
          topkIndicesHostData[i] = i;
      }
  
    // 创建query aclTensor
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建key aclTensor
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建value aclTensor
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建attentionOut aclTensor
      ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建attentionOutGrad aclTensor
      ret = CreateAclTensor(attentionOutGradHostData, attentionOutGradShape, &attentionOutGradDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutGrad);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建softmaxMax aclTensor
      ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建softmaxSum aclTensor
      ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建topkIndices aclTensor
      ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndices);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int64_t tempQ[1] = {1};
      int64_t tempK[1] = {1024};
      aclIntArray* actualSeqQLenOptional = aclCreateIntArray(tempQ, static_cast<uint64_t>(1));
      aclIntArray* actualSeqKvLenOptional = aclCreateIntArray(tempK, static_cast<uint64_t>(1));
      // 创建dq aclTensor
      ret = CreateAclTensor(dqOutHostData, dqOutShape, &dqOutDeviceAddr, aclDataType::ACL_FLOAT16, &dqOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dk aclTensor
      ret = CreateAclTensor(dkOutHostData, dkOutShape, &dkOutDeviceAddr, aclDataType::ACL_FLOAT16, &dkOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dv aclTensor
      ret = CreateAclTensor(dvOutHostData, dvOutShape, &dvOutDeviceAddr, aclDataType::ACL_FLOAT16, &dvOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  
    // aclnnNsaSelectedAttentionGrad接口调用示例
      // 3. 调用CANN算子库API，需要修改为具体的API名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // 调用aclnnNsaSelectedAttentionGrad第一段接口
      ret = aclnnNsaSelectedAttentionGradGetWorkspaceSize(query, key, value, attentionOut, attentionOutGrad, softmaxMax,
                                                          softmaxSum, topkIndices, actualSeqQLenOptional,
                                                          actualSeqKvLenOptional, nullptr, scaleValue, selectedBlockSize,
                                                          selectedBlockCount, headNum, inputLayout, sparseMode,
                                                          dqOut, dkOut, dvOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnNsaSelectedAttentionGrad第二段接口
      ret = aclnnNsaSelectedAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGrad failed. ERROR: %d\n", ret); return ret);
  
    // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
      auto dqSize = GetShapeSize(dqOutShape);
      std::vector<aclFloat16> dqResultData(dqSize, 0);
      ret = aclrtMemcpy(dqResultData.data(), dqResultData.size() * sizeof(dqResultData[0]), dqOutDeviceAddr,
                        dqSize * sizeof(dqResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dq from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < dqSize; i++) {
          LOG_PRINT("result dq[%ld] is: %f\n", i, dqResultData[i]);
      }
  
    auto dkSize = GetShapeSize(dkOutShape);
      std::vector<aclFloat16> dkResultData(dkSize, 0);
      ret = aclrtMemcpy(dkResultData.data(), dkResultData.size() * sizeof(dkResultData[0]), dkOutDeviceAddr,
                        dkSize * sizeof(dkResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dk from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < dkSize; i++) {
          LOG_PRINT("result dk[%ld] is: %f\n", i, dkResultData[i]);
      }
  
    auto dvSize = GetShapeSize(dvOutShape);
      std::vector<aclFloat16> dvResultData(dkSize, 0);
      ret = aclrtMemcpy(dvResultData.data(), dvResultData.size() * sizeof(dvResultData[0]), dkOutDeviceAddr,
                        dvSize * sizeof(dvResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dv from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < dvSize; i++) {
          LOG_PRINT("result dv[%ld] is: %f\n", i, dkResultData[i]);
      }
  
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(query);
      aclDestroyTensor(key);
      aclDestroyTensor(value);
      aclDestroyTensor(attentionOut);
      aclDestroyTensor(attentionOutGrad);
      aclDestroyTensor(softmaxMax);
      aclDestroyTensor(softmaxSum);
      aclDestroyTensor(topkIndices);
      aclDestroyTensor(dqOut);
      aclDestroyTensor(dkOut);
      aclDestroyTensor(dvOut);
      aclDestroyIntArray(actualSeqQLenOptional);
      aclDestroyIntArray(actualSeqKvLenOptional);
      // 7. 释放device资源，需要根据具体API的接口定义修改
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(attentionOutDeviceAddr);
      aclrtFree(attentionOutGradDeviceAddr);
      aclrtFree(softmaxMaxDeviceAddr);
      aclrtFree(softmaxSumDeviceAddr);
      aclrtFree(topkIndicesDeviceAddr);
      aclrtFree(dqOutDeviceAddr);
      aclrtFree(dkOutDeviceAddr);
      aclrtFree(dvOutDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  
```