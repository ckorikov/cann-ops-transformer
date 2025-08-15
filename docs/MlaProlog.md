声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MlaProlog

## 支持的产品型号
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 功能说明
-  **算子功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分为四路，首先对输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后分为两路，第一路乘以$W^{UQ}$和$W^{UK}$经过两次上采样后得到$q^N$；第二路乘以$W^{QR}$后经过旋转位置编码（ROPE）得到$q^R$；第三路是输入$x$乘以$W^{DKV}$进行下采样和RmsNorm后传入Cache中得到$k^C$；第四路是输入$x$乘以$W^{KR}$后经过旋转位置编码后传入另一个Cache中得到$k^R$。
-  **计算公式**：

    RmsNorm公式

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query计算公式，包括下采样，RmsNorm和两次上采样

    $$
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    对Query的进行ROPE旋转位置编码

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    Key计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$

## 实现原理

详细实现原理参考[MlaProlog算子设计介绍](./common/MlaProlog算子设计介绍.md)。

## 算子执行接口
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMlaPrologGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnMlaProlog”接口执行计算。

* `aclnnStatus aclnnMlaPrologGetWorkspaceSize(const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cacheIndex, aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional, const aclTensor *queryOut, const aclTensor *queryRopeOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMlaProlog(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMlaPrologGetWorkspaceSize
```
shape格式字段含义：
  - B：Batch 表示输入样本批量大小，取值范围：0~65536
  - S：Seq-Length 表示输入样本序列长度，取值范围：0~16
  - He：Head-Size 表示隐藏层的大小，取值固定为：7168
  - Hcq：q低秩矩阵维度，取值固定为：1536
  - N：Head-Num 表示多头数，取值范围：1、2、4、8、16、32、64、128
  - Hckv：kv低秩矩阵维度，取值固定为：512
  - D：qk不含位置编码维度，取值固定为：128
  - Dr：qk位置编码维度，取值固定为：64
  - Nkv：kv的head数，取值固定为：1
  - BlockNum：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度，该值允许取0。
  - BlockSize：PagedAttention场景下的块大小，取值范围：16、128
  - T：BS合轴后的大小，取值范围：0~1048576。注：若采用BS合轴，此时tokenX、ropeSin、ropeCos均为2维，cacheIndex为1维，queryOut、queryRopeOut为3维。
```

- **参数说明：**

  - tokenX（aclTensor\*，计算输入）：表示输入的tensor，用于计算Query和Key的x，Device侧的aclTensor。shape支持2维和3维，格式为(T,He)和(B,S,He)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - weightDq（aclTensor\*，计算输入）：表示用于计算Query的下采样权重矩阵，对应公式中的$W^{DQ}$，Device侧的aclTensor。其shape支持2维，格式为(He,Hcq)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持FRACTAL_NZ格式。
  - weightUqQr（aclTensor\*，计算输入）：表示用于计算Query的上采样权重矩阵和Query的位置编码权重矩阵，对应公式中的$W^{UQ}$和$W^{QR}$，Device侧的aclTensor。其shape支持2维，格式为(Hcq,N*(D+Dr))，dtype支持BFLOAT16和INT8，[数据格式](common/数据格式.md)支持FRACTAL_NZ格式。
    - 当weightUqQr为INT8类型时，weightUqQr是一个per-tensor的量化后的输入，表示当前为部分量化场景。
      - 此时若kvCacheRef、krCacheRef为BFLOAT16类型，对应输出为非量化输出，此时dequantScaleWUqQrOptional字段必须传入，smoothScalesCqOptional字段可选传入。
      - 此时若kvCacheRef、krCacheRef为INT8类型，对应输出为量化输出，此时dequantScaleWUqQrOptional、quantScaleCkvOptional、quantScaleCkrOptional字段必须传入，smoothScalesCqOptional字段可选传入。
    - 当weightUqQr为BFLOAT16类型时，表示当前为非量化场景。
      - 此时dequantScaleWUqQrOptional、quantScaleCkvOptional、quantScaleCkrOptional、smoothScalesCqOptional字段不能传入（即为nullptr）。
  - weightUk（aclTensor\*，计算输入）：表示用于计算Key的上采样权重，对应公式中的$W^{UK}$，Device侧的aclTensor。其shape支持3维，格式为(N,D,Hckv)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - weightDkvKr（aclTensor\*，计算输入）：表示用于计算Key的下采样权重矩阵和Key的位置编码权重矩阵，对应公式中的$W^{DKV}$和$W^{KR}$，Device侧的aclTensor。其shape支持2维，格式为(He,Hckv+Dr)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持FRACTAL_NZ格式。
  - rmsnormGammaCq（aclTensor\*，计算输入）：表示计算$c^Q$的RmsNorm公式中的$\gamma$参数，Device侧的aclTensor。其shape支持1维，格式为(Hcq)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - rmsnormGammaCkv（aclTensor\*，计算输入）：表示计算$c^{KV}$的RmsNorm公式中的$\gamma$参数，Device侧的aclTensor。其shape支持1维，格式为(Hckv)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - ropeSin（aclTensor\*，计算输入）：表示用于计算旋转位置编码的正弦参数矩阵，Device侧的aclTensor。其shape支持2维和3维，格式为(T,Dr)和(B,S,Dr)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - ropeCos（aclTensor\*，计算输入）：表示用于计算旋转位置编码的余弦参数矩阵，Device侧的aclTensor。其shape支持2维和3维，格式为(T,Dr)和(B,S,Dr)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - cacheIndex（aclTensor\*，计算输入）：表示用于存储kvCache和krCache的索引，Device侧的aclTensor。其shape支持1维和2维，格式为(T)和(B,S)，dtype支持INT64，[数据格式](common/数据格式.md)支持ND格式。
    - cacheIndex的取值范围为[0,BlockNum*BlockSize)，当前不会对cacheIndex传入值的合法性进行校验，需用户自行保证。
  - kvCacheRef（aclTensor\*，计算输入）：表示用于cache索引的aclTensor。其shape支持4维，格式为(BlockNum,BlockSize,Nkv,Hckv)，dtype支持BFLOAT16和INT8，[数据格式](common/数据格式.md)支持ND格式。**计算结果原地更新**，更新后结果对应公式中的$k^C$。
  - krCacheRef（aclTensor\*，计算输入）：表示用于key位置编码的cache，Device侧的aclTensor。其shape支持4维，格式为(BlockNum,BlockSize,Nkv,Dr)，dtype支持BFLOAT16和INT8，[数据格式](common/数据格式.md)支持ND格式。**计算结果原地更新**，更新后结果对应公式中的$k^R$。
  - dequantScaleXOptional（aclTensor\*，计算输入）：**预留参数，当前版本暂未使用，传入空指针**。
  - dequantScaleWDqOptional（aclTensor\*，计算输入）：**预留参数，当前版本暂未使用，传入空指针**。
  - dequantScaleWUqQrOptional（aclTensor\*，计算输入）：用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化参数为per-channel，Device侧的aclTensor。其shape支持2维，格式为(1,N*(D+Dr))，dtype支持FLOAT，[数据格式](common/数据格式.md)支持ND格式。
  - dequantScaleWDkvKrOptional（aclTensor\*，计算输入）：**预留参数，当前版本暂未使用，传入空指针**。
  - quantScaleCkvOptional（aclTensor\*，计算输入）：用于对输出到KVCache中的数据做量化操作时的参数，Device侧的aclTensor。其shape支持2维，格式为(1,Hckv)，dtype支持FLOAT，[数据格式](common/数据格式.md)支持ND格式。
  - quantScaleCkrOptional（aclTensor\*，计算输入）：用于对输出到KRCache中的数据做量化操作时的参数，Device侧的aclTensor。其shape支持2维，格式为(1,Dr)，dtype支持FLOAT，[数据格式](common/数据格式.md)支持ND格式。
  - smoothScalesCqOptional（aclTensor\*，计算输入）：用于对RmsNormCq输出做动态量化操作时的参数，Device侧的aclTensor。其shape支持2维，格式为(1,Hcq)，dtype支持FLOAT，[数据格式](common/数据格式.md)支持ND格式。
  - rmsnormEpsilonCq（double，计算输入）：表示计算$c^Q$的RmsNorm公式中的$\epsilon$参数，用户不特意指定时建议传入1e-05。
  - rmsnormEpsilonCkv（double，计算输入）：表示计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数，用户不特意指定时建议传入1e-05。
  - cacheModeOptional（char\*，计算输入）：表示kvCache的模式，支持"PA_BSND"，"PA_NZ"，用户不特意指定时建议传入"PA_BSND"。
  - queryOut（aclTensor\*，计算输出）：表示Query的输出tensor，对应公式中的$q^N$，Device侧的aclTensor。shape支持3维和4维，格式为(T,N,Hckv)和(B,S,N,Hckv)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - queryRopeOut（aclTensor\*，计算输出）：表示Query位置编码的输出tensor，对应公式中的$q^R$，Device侧的aclTensor。shape支持3维和4维，格式为(T,N,Dr)和(B,S,N,Dr)，dtype支持BFLOAT16，[数据格式](common/数据格式.md)支持ND格式。
  - workspaceSize（uint64_t\*，计算输出）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，计算输出）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：必须传入的参数中存在空指针。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：输入参数的shape、dtype和数据类型不在支持的范围之内。
  - 返回361001（ACLNN_ERR_RUNTIME_ERROR）：API内存调用npu runtime的接口异常。
  ```

### aclnnMlaProlog

- **参数说明：**

  * workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMlaPrologGetWorkspaceSize获取。
  * executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
    - 如果B、S、T取值为0，则queryOut、queryRopeOut输出空Tensor，kvCacheRef、krCacheRef不做更新。
    - 如果Skv取值为0，则queryOut、queryRopeOut正常计算，kvCacheRef、krCacheRef不做更新，即输出空Tensor。

## 算子原型
```
REG_OP(MlaProlog)
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cache_index, TensorType({DT_INT64}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)
    .ATTR(cache_mode, String, "PA_BSND")
    .OP_END_FACTORY_REG(MlaProlog)
```


参数解释请参见**算子执行接口**。

## 调用示例

该融合算子有两种调用方式：

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_mla_prolog](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mla_prolog.h"
  
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
      int64_t shape_size = 1;
      for (auto i : shape) {
          shape_size *= i;
      }
      return shape_size;
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
  int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMalloc申请host侧内存
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  template <typename T>
  int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMalloc申请host侧内存
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  int TransToNZShape(std::vector<int64_t> &shapeND) {
      int64_t inputParam1 = shapeND[0];
      int64_t inputParam2 = shapeND[1];
      int64_t h0 = 16;
      int64_t newParam1 = inputParam2 / h0;
      int64_t newParam2 = inputParam1 / h0;
      shapeND[0] = newParam1;
      shapeND[1] = newParam2;
      shapeND.emplace_back(h0);
      shapeND.emplace_back(h0);
      return 0;
  }
  
  int main() {
      // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. 构造输入与输出，需要根据API的接口定义构造
      std::vector<int64_t> tokenXShape = {8, 1, 7168};  // B,S,He
      std::vector<int64_t> weightDqShape = {7168, 1536};  // He,Hcq
      std::vector<int64_t> weightUqQrShape = {1536, 6144};  // Hcq,N*(D+Dr)
      std::vector<int64_t> weightUkShape = {32, 128, 512};  // N,D,Hckv
      std::vector<int64_t> weightDkvKrShape = {7168, 576};  // He,Hckv+Dr
      std::vector<int64_t> rmsnormGammaCqShape = {1536};  // Hcq
      std::vector<int64_t> rmsnormGammaCkvShape = {512};  // Hckv
      std::vector<int64_t> ropeSinShape = {8, 1, 64};  // B,S,Dr
      std::vector<int64_t> ropeCosShape = {8, 1, 64};  // B,S,Dr
      std::vector<int64_t> cacheIndexShape = {8, 1};  // B,S
      std::vector<int64_t> kvCacheShape = {16, 128, 1, 512};  // BolckNum,BlockSize,Nkv,Hckv
      std::vector<int64_t> krCacheShape = {16, 128, 1, 64};  // BolckNum,BlockSize,Nkv,Dr
      std::vector<int64_t> queryShape = {8, 1, 32, 512};  // B,S,N,Hckv
      std::vector<int64_t> queryRopeShape = {8, 1, 32, 64};  // B,S,N,Dr
      double rmsnormEpsilonCq = 1e-5;
      double rmsnormEpsilonCkv = 1e-5;
      char cacheMode[] = "PA_BSND";
  
      void* tokenXDeviceAddr = nullptr;
      void* weightDqDeviceAddr = nullptr;
      void* weightUqQrDeviceAddr = nullptr;
      void* weightUkDeviceAddr = nullptr;
      void* weightDkvKrDeviceAddr = nullptr;
      void* rmsnormGammaCqDeviceAddr = nullptr;
      void* rmsnormGammaCkvDeviceAddr = nullptr;
      void* ropeSinDeviceAddr = nullptr;
      void* ropeCosDeviceAddr = nullptr;
      void* cacheIndexDeviceAddr = nullptr;
      void* kvCacheDeviceAddr = nullptr;
      void* krCacheDeviceAddr = nullptr;
      void* queryDeviceAddr = nullptr;
      void* queryRopeDeviceAddr = nullptr;
  
      void* tokenXHostAddr = nullptr;
      void* weightDqHostAddr = nullptr;
      void* weightUqQrHostAddr = nullptr;
      void* weightUkHostAddr = nullptr;
      void* weightDkvKrHostAddr = nullptr;
      void* rmsnormGammaCqHostAddr = nullptr;
      void* rmsnormGammaCkvHostAddr = nullptr;
      void* ropeSinHostAddr = nullptr;
      void* ropeCosHostAddr = nullptr;
      void* cacheIndexHostAddr = nullptr;
      void* kvCacheHostAddr = nullptr;
      void* krCacheHostAddr = nullptr;
      void* queryHostAddr = nullptr;
      void* queryRopeHostAddr = nullptr;
  
      aclTensor* tokenX = nullptr;
      aclTensor* weightDq = nullptr;
      aclTensor* weightUqQr = nullptr;
      aclTensor* weightUk = nullptr;
      aclTensor* weightDkvKr = nullptr;
      aclTensor* rmsnormGammaCq = nullptr;
      aclTensor* rmsnormGammaCkv = nullptr;
      aclTensor* ropeSin = nullptr;
      aclTensor* ropeCos = nullptr;
      aclTensor* cacheIndex = nullptr;
      aclTensor* kvCache = nullptr;
      aclTensor* krCache = nullptr;
      aclTensor* query = nullptr;
      aclTensor* queryRope = nullptr;
  
      // 转换三个NZ格式变量的shape
      ret = TransToNZShape(weightDqShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
  
      // 创建tokenX aclTensor
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_BF16, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightDq aclTensor
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_BF16, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightUqQr aclTensor
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_BF16, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightUk aclTensor
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightDkvKr aclTensor
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_BF16, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建ropeSin aclTensor
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建ropeCos aclTensor
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rmsnormGammaCq aclTensor
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rmsnormGammaCkv aclTensor
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建cacheIndex aclTensor
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建kvCache aclTensor
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_BF16, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建krCache aclTensor
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建query aclTensor
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_BF16, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建queryRope aclTensor
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  
      // 3. 调用CANN算子库API，需要修改为具体的API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // 调用aclnnMlaProlog第一段接口
      ret = aclnnMlaPrologGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, cacheIndex, kvCache, krCache, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode, query, queryRope, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      // 调用aclnnMlaProlog第二段接口
      ret = aclnnMlaProlog(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaProlog failed. ERROR: %d\n", ret); return ret);
  
      // 4. 固定写法，同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  
      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(tokenX);
      aclDestroyTensor(weightDq);
      aclDestroyTensor(weightUqQr);
      aclDestroyTensor(weightUk);
      aclDestroyTensor(weightDkvKr);
      aclDestroyTensor(rmsnormGammaCq);
      aclDestroyTensor(rmsnormGammaCkv);
      aclDestroyTensor(ropeSin);
      aclDestroyTensor(ropeCos);
      aclDestroyTensor(cacheIndex);
      aclDestroyTensor(kvCache);
      aclDestroyTensor(krCache);
      aclDestroyTensor(query);
      aclDestroyTensor(queryRope);
  
      // 7. 释放device 资源
      aclrtFree(tokenXDeviceAddr);
      aclrtFree(weightDqDeviceAddr);
      aclrtFree(weightUqQrDeviceAddr);
      aclrtFree(weightUkDeviceAddr);
      aclrtFree(weightDkvKrDeviceAddr);
      aclrtFree(rmsnormGammaCqDeviceAddr);
      aclrtFree(rmsnormGammaCkvDeviceAddr);
      aclrtFree(ropeSinDeviceAddr);
      aclrtFree(ropeCosDeviceAddr);
      aclrtFree(cacheIndexDeviceAddr);
      aclrtFree(kvCacheDeviceAddr);
      aclrtFree(krCacheDeviceAddr);
      aclrtFree(queryDeviceAddr);
      aclrtFree(queryRopeDeviceAddr);
  
      // 8. 释放host 资源
      aclrtFree(tokenXHostAddr);
      aclrtFree(weightDqHostAddr);
      aclrtFree(weightUqQrHostAddr);
      aclrtFree(weightUkHostAddr);
      aclrtFree(weightDkvKrHostAddr);
      aclrtFree(rmsnormGammaCqHostAddr);
      aclrtFree(rmsnormGammaCkvHostAddr);
      aclrtFree(ropeSinHostAddr);
      aclrtFree(ropeCosHostAddr);
      aclrtFree(cacheIndexHostAddr);
      aclrtFree(kvCacheHostAddr);
      aclrtFree(krCacheHostAddr);
      aclrtFree(queryHostAddr);
      aclrtFree(queryRopeHostAddr);
  
      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  
      return 0;
  }
  ```



