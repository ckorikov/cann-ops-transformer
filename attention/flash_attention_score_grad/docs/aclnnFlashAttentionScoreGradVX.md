# FlashAttentionScoreGradVX


## 产品支持情况
|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200/300/500 推理产品</term>|      ×     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：训练场景下计算注意力的反向输出，即[FlashAttentionScoreVX](./FlashAttentionScoreVX.md)的反向计算。**该接口query、key、value参数支持多个长度相等或者长度不相等的sequence**
  - **该接口合并了[FlashAttentionScoreGradV2](./FlashAttentionScoreGradV2.md)接口和[FlashAttentionUnpaddingScoreGradV2](./FlashAttentionUnpaddingScoreGradV2.md)接口，并调整了Dropout功能**：
    -   <term>昇腾910_95 AI处理器</term>：keepProb小于1.0时，若没有外部传入的DropoutMask，则使用新增参数生成DropoutMask；若有外部传入的DropoutMask，则使用外部传入的DropoutMask
  
- 计算公式：
  - pseType=1时，与[FlashAttentionScoreGrad](./FlashAttentionScoreGrad.md)计算公式相同
  - pseType=其他取值时，公式如下：

  $$
  Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
  $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$
  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$
  $$
  Y=PV
  $$

  则注意力的反向计算公式为：

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

  **说明：**
  query、keyIn、value数据排布格式支持从多种维度解读，其中T (Total S Length) 表示所有batch对应的S的总长、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、d（Head-Dim）表示隐藏层最小的单元尺寸，且满足d=H/N。

## 实现原理

实现原理同[FlashAttentionScoreGradV2](./FlashAttentionScoreGradV2.md)。

## 算子执行接口

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFlashAttentionScoreGradVXGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionScoreGradVX”接口执行计算。

* `aclnnStatus aclnnFlashAttentionScoreGradVXGetWorkspaceSize(const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy, const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional, const aclTensor *dScaleOOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValueOptional, double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum, char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,int64_t seedOptional, int64_t offsetOptional, int64_t outDtypeOptional, aclTensor *dqOut, aclTensor *dkOut, aclTensor *dvOut, aclTensor *dqRopeOut, aclTensor *dkRopeOut, aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnFlashAttentionScoreGradVX(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnFlashAttentionScoreGradVXGetWorkspaceSize

>**说明：**
>query、keyIn、value数据排布格式支持从多种维度解读，其中T (Total S Length) 表示所有batch对应的S的总长、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、d（Head-Dim）表示隐藏层最小的单元尺寸，且满足d=H/N。

- **参数说明：**
  - query（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入Q，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32

  - queryRopeOptional（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入Q的rope部分，即旋转位置编码，数据类型支持BFLOAT16和FLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。

  - keyIn（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入K，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32

  - keyRopeOptional（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入K的rope部分，数据类型支持BFLOAT16和FLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。

  - value（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入V，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32

  - dy（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入dY，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32

  - pseShiftOptional（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入pse，可选参数，表示位置编码，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持shape范围为\[B,N,H,S\]、\[1,N,H,S\]，H固定为1024；alibi位置编码场景，preTokens和nextTokens必须配置下三角；如果pseType为2或3的时候，数据类型需为FLOAT32, 对应shape支持范围是\[B,N],\[N]。

  - dropMaskOptional（aclTensor\*，计算输入）：Device侧的aclTensor，数据类型支持UINT8，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。如不使用该参数，可传入nullptr。其shape和数据排布可表示为：

    $$
    (\sum_{b=0}^{B-1} (\sum_{n=0}^{N-1}(Sq*Skv)))/8
    $$

  - paddingMaskOptional（aclTensor\*，计算输入）：Device侧的aclTensor，**预留参数暂未使用，调用时该参数需传空**。

  - qStartIdxOptional（aclIntArray\*，计算输入）：Host侧的aclIntArray，可选参数，数据类型支持INT64，代表外切场景，当前分块的Q的sequence在全局中的起始索引，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - kvStartIdxOptional（aclIntArray\*，计算输入）：Host侧的aclIntArray，可选参数，数据类型支持INT64，代表外切场景，当前分块的Q的sequence在全局中的起始索引，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - attenMaskOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选属性，数据类型支持BOOL\(8bit的BOOL\)、UINT8，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持shape范围为\[S1Max, S2Max\]；综合约束请见[约束说明](#1)。

  - softmaxMaxOptional（aclTensor\*，计算输入）：Device侧的aclTensor，注意力正向计算的中间输出，数据类型支持FLOAT，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。

  - softmaxSumOptional（aclTensor\*，计算输入）：Device侧的aclTensor，注意力正向计算的中间输出，数据类型支持FLOAT，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。综合约束请见[约束说明](#1)。

  - softmaxInOptional（aclTensor\*，计算输入）：Device侧的aclTensor，注意力正向计算的中间输出，**预留参数暂未使用，调用时该参数需传空**。

  - attentionInOptional（aclTensor\*，计算输入）：Device侧的aclTensor，注意力正向计算的最终输出，数据类型和shape与query一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32

  - dScaleQOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，是query输入的反量化参数，数据类型支持FLOAT32，数据类型为[B,N2,G,S1/128,1]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - dScaleKOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，是key输入的反量化参数，数据类型支持FLOAT32，数据类型为[B,N2,1,S2/128,1]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - dScaleVOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，是value输入的反量化参数，数据类型支持FLOAT32，数据类型为[B,N2,1,S2/128,1]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - dScaleDyOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，是dy输入的反量化参数，数据类型支持FLOAT32，数据类型为[B,N2,G,S1/128,1]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - dScaleOOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选参数，是attentionOptional输入的反量化参数，数据类型支持FLOAT32，数据类型为[B,N2,G,S1/128,1]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - prefixOptional（aclTensor\*，计算输入）：Device侧的aclTensor，可选属性，代表prefix稀疏计算场景每个Batch的N值，数据类型支持INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND；综合约束请见[约束说明](#1)。

  - actualSeqQLenOptional（aclIntArray\*，计算输入）：数据类型支持INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，描述了每个Batch对应的query S大小；综合约束请见[约束说明](#1)。

  - actualSeqKvLenOptional（aclIntArray\*，计算输入）：数据类型支持INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，描述了每个Batch对应的key/value S大小。

  - scaleValueOptional（double，计算输入）：Host侧的double，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。一般设置为d^-0.5。

  - keepProbOptional（double，计算输入）：Host侧的double，代表dropMaskOptional中1的比例，数据类型支持DOUBLE；综合约束请见[约束说明](#1)。

  - preTokensOptional（int64\_t，计算输入）：Host侧的int64\_t，用于稀疏计算的参数，数据类型支持INT64。

  - nextTokensOptional（int64\_t，计算输入）：Host侧的int64\_t，用于稀疏计算的参数，数据类型支持INT64。

  - headNum（int64\_t，计算输入）：Host侧的int64\_t，代表head个数，数据类型支持INT64；综合约束请见[约束说明](#1)。

  - inputLayout（char\*，计算输入）：Host侧的string，代表输入query、keyIn、value的数据排布格式，支持BSH、SBH、BSND、BNSD、TND。

  - innerPreciseOptional（int32\_t，计算输入）：**预留参数暂未使用，调用时该参数需传空**。

  - sparseModeOptional（int64\_t，计算输入）：Host侧的int，表示sparse的模式，数据类型支持INT64。

    -   sparseModeOptional为0时，代表defaultMask模式，如果attenMaskOptional未传入则不做mask操作，忽略preTokensOptional和nextTokensOptional\(内部赋值为INT\_MAX\)；如果传入，则需要传入完整的attenMaskOptional矩阵（S1Max \* S2Max），表示preTokensOptional和nextTokensOptional之间的部分需要计算。
    -   sparseModeOptional为1时，代表allMask，即传入完整的attenMaskOptional矩阵。
    -   sparseModeOptional为2时，代表leftUpCausal模式的mask，对应以左顶点为划分的下三角场景，需要传入优化后的attenMaskOptional矩阵（2048\*2048）。
    -   sparseModeOptional为3时，代表rightDownCausal模式的mask，对应以右下顶点为划分的下三角场景，需要传入优化后的attenMaskOptional矩阵（2048\*2048）。
    -   sparseModeOptional为4时，代表band场景，即计算preTokensOptional和nextTokensOptional之间的部分。
    -   sparseModeOptional为5时，代表prefix场景，即在rightDownCausal的基础上，左侧加上一个长为S1，宽为N的矩阵，N的值由输入。
    -   sparseModeOptional为6时，代表prefix压缩场景，需要传入shape为\[3072, 2048\]的attenMaskOptional矩阵；分为两部分：其中上半部分为\[2048, 2048\]的下三角矩阵；下半部分为\[1024, 2048\]的矩阵，矩形矩阵左半部分全0，右半部分全1。0代表保留，1代表掩掉。
    -   sparseModeOptional为7时，代表rightDownCausal_Band场景，该场景由长序列外切产生，需要正确配置preTokensOptional和nextTokensOptional参数；传入shape为\[2048, 2048\]的下三角attenMaskOptional矩阵。
    -   sparseModeOptional为8时，代表band_LeftUpCausal场景，该场景由长序列外切产生，需要正确配置preTokensOptional和nextTokensOptional参数；传入shape为\[2048, 2048\]的下三角attenMaskOptional矩阵。

    用户不特意指定时建议传入0。sparse不同模式的详细说明请参见[sparse模式说明](./common/sparse_mode参数说明.md)。

    >**说明：**
    >当所有的attenMaskOptional的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量；sparseModeOptional配置为1、2、3、5、6时，用户配置的preTokensOptional、nextTokensOptional不会生效；sparseModeOptional配置为0、4时，须保证attenMaskOptional与preTokensOptional、nextTokensOptional的范围一致。
    >layout为非TND时支持sparseModeOptional配置范围：0到6（包含0和6），layout为TND时支持sparseModeOptional配置范围：0到8（包含0和8），但不支持值5。

  - pseTypeOptional （int64\_t，计算输入）：Host侧的整型，数据类型支持INT64，用户不特意指定时可传入1，跟当前[FlashAttentionScoreGrad](./FlashAttentionScoreGrad.md)实现一致，支持配置值为0、1、2、3。
    | pseType     | 含义                              |      备注   |
    | ----------- | --------------------------------- | ----------|
    | 0           | 外部传入pse 先mul再add              | - |
    | 1           | 外部传入pse 先add再mul              | 跟[FlashAttentionScoreGrad](./FlashAttentionScoreGrad.md)实现一致。 |
    | 2           | 内部生成pse 先mul再add              | - |
    | 3           | 内部生成pse 先mul再add再sqrt         | - |

  - seedOptional（int64\_t，计算输入）：Host侧的整型。数据类型支持INT64。keepProbOptional小于1.0时，根据seedOptional和offsetOptional生成DropoutMask。
    
  - offsetOptional（int64\_t，计算输入）：Host侧的整型。数据类型支持INT64。
  
  - outDtypeOptional（int64\_t，计算输入）：Host侧的整型。值为0表示dqOut等输出是FLOAT16类型，值为1表示dqOut等输出是BFLOAT16格式。
    
  - dqOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的dQ，表示query的梯度，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT32
      -   如果query数据类型为FLOAT8_E5M2或FLOAT8_E4M3FN，则数据类型根据outDtypeOptional决定
        -   如果query数据类型为FLOAT16或BFLOAT16或FLOAT32，则数据类型与query数据类型一致
  
  - dqRopeOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的dqRope，表示queryRope的梯度，计算输出，数据类型支持BFLOAT16和FLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  - dkOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的dK，表示keyIn的梯度，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT32
      -   如果query数据类型为FLOAT8_E5M2或FLOAT8_E4M3FN，则数据类型根据outDtypeOptional决定
        -   如果query数据类型为FLOAT16或BFLOAT16或FLOAT32，则数据类型与query数据类型一致
  
  - dkRopeOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的dkRope，表示keyInRope的梯度，计算输出，数据类型支持BFLOAT16和FLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。  

  - dvOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的dV，表示value的梯度，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT32
      -   如果query数据类型为FLOAT8_E5M2或FLOAT8_E4M3FN，则数据类型根据outDtypeOptional决定
        -   如果query数据类型为FLOAT16或BFLOAT16或FLOAT32，则数据类型与query数据类型一致
  
  - dpseOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的d\(pse\)，表示pse的梯度，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，**预留参数暂未使用，调用时该参数需传空**。但在pseShiftOptional不为空时，shape和数据类型与pseShiftOptional一致。
    -   <term>昇腾910_95 AI处理器</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT32
      -   如果query数据类型为FLOAT8_E5M2或FLOAT8_E4M3FN，则数据类型根据outDtypeOptional决定
      -   如果query数据类型为FLOAT16或BFLOAT16或FLOAT32，则数据类型与query数据类型一致
  
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
  
- **返回值：**

     返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

     ```
     第一段接口完成入参校验，若出现以下错误码，则对应原因为：
     - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
     - 返回161002（ACLNN_ERR_PARAM_INVALID）：query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、dqOut、dkOut、dvOut的数据类型和数据格式不在支持的范围内。
     ```

### aclnnFlashAttentionScoreGradVX

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionScoreGradVXGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的Stream。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明<a name="1"></a>

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 输入query、key、value、dy的B：batchsize必须相等。
- 输入query、key、value、dy的input_layout必须一致。
- 输入query、key、value、dy的D：Head-Dim必须满足query和key的D相等，value和dy的D相等，并且query和key的D大于等于value和dy的D。
- 支持输入query/dy的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。
- 关于数据shape的约束，以inputLayout的TND为例，其中：

    -   B：取值范围为1\~2K。带prefixOptional的时候B最大支持1K。
    -   N：取值范围为1\~256。
    -   S：取值范围为1\~1M。
    -   D：取值范围为1\~512。
    -   KeepProb：取值范围为(0, 1]。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- prefixOptional稀疏计算仅支持压缩场景，sparseModeOptional=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。
- sparse_mode=7时，不支持可选输入realShiftOptional。
- sparse_mode=8时，当每个sequence的q、kv等长时支持可选输入realShiftOptional，针对全局做pse生成。支持q方向进行外切，需要外切前每个sequence的q、kv等长，外切后传入的actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0（本功能属实验性功能）。
- actualSeqQLenOptional输入支持某个Batch上的S长度为0，此时不支持可选输入pseShiftOptional。
- 关于softmaxMax与softmaxSum参数的约束：输入格式固定为\[B, N, S, 8\],TND的输入格式除外，此时为\[T, N, 8\]，注：T=B*S。
- headNum的取值必须和传入的Query中的N值保持一致。
- <term>昇腾910_95 AI处理器</term>：

    -   seedOptional和offsetOptional只在keepProbOptional小于1.0时生效，否则不生效。
    -   keepProbOptional小于1.0时，若dropMaskOptional非nullptr，则使用输入的dropMask；否则使用seed和offset生成的dropMask。

## 算子原型

```c++
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_dy, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_o, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dq_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .ATTR(seed, Int, 0)
    .ATTR(offset, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)
```
参数解释请参见**算子执行接口**。


## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
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
  // 1. （固定写法）device/stream初始化，acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};

  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dvShape = {256, 1, 128};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dScaleO = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;

  std::vector<short> qHostData(32768, 1);
  std::vector<short> kHostData(32768, 1);
  std::vector<short> vHostData(32768, 1);
  std::vector<short> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<short> attentionInHostData(32768, 1);
  std::vector<short> dqHostData(32768, 0);
  std::vector<short> dkHostData(32768, 0);
  std::vector<short> dvHostData(32768, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> prefixOp = {0};
  aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t>  acSeqQLenOp = {256};
  std::vector<int64_t>  acSeqKvLenOp = {256};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMod = 0;
  int64_t pseType = 1;
  char layOut[5] = {'T', 'N', 'D', 0};
  
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // 调用aclnnFlashAttentionScoreGradVX第一段接口
  ret = aclnnFlashAttentionScoreGradVXGetWorkspaceSize(q, k, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, dScaleQ, dScaleK, dScaleV,
            dScaleDy, dScaleO, prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMod, pseType,
            0, 0, 0, dq, dk, dv, dpse, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradVXGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnFlashAttentionScoreGrad第二段接口
  ret = aclnnFlashAttentionScoreGradVX(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradVX failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}
```