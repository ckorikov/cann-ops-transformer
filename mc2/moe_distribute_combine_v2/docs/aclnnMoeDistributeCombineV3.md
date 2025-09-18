# aclnnMoeDistributeCombineV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。
$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
$$

> 注意：该接口必须与`aclnnMoeDistributeDispatchV3`配套使用，相当于按`aclnnMoeDistributeDispatchV3`接口收集数据的路径原路返还。

相较于`aclnnMoeDistributeCombineV2`接口，该接口变更如下：
- 新增支持动态缩容场景：支持在创建通信域后，剔除故障卡，算子可正常执行（无需重新编译），通过传入`elasticInfoOptional`参数使能该特性。
- 新增支持零专家场景：
  - **zeroExpert**：`Moe(x) = 0`，通过传入大于0的`zeroExpertNum`参数使能。
  - **copyExpert**：`Moe(x) = x`，通过传入大于0的`copyExpertNum`参数使能，且需传入有效的`oriXOptional`参数。
  - **constExpert**：`Moe(x) = alpha * x + alpha2 * v`，通过传入大于0的`constExpertNum`参数使能，且需传入有效的`oriXOptional`、`constExpertAlpha1Optional`、`constExpertAlpha2Optional`、`constExpertVOptional`参数。


## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnMoeDistributeCombineV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeCombineV3”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeCombineV3GetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
    const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional,
    const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional,
    const aclTensor* elasticInfoOptional,
    const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional,
    const aclTensor* constExpertAlpha2Optional,
    const aclTensor* constExpertVOptional,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    const char* groupTp,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t outDtype,
    int64_t commQuantMode,
    int64_t groupListType,
    const char* commAlg,
    int64_t zeroExpertNum,
    int64_t copyExpertNum,
    int64_t constExpertNum,
    aclTensor* xOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineV3(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnMoeDistributeCombineV3GetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"> 
<colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 400px">
 <col style="width: 200px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>expandX</td>
   <td>输入</td>
   <td>根据expertIds扩展的token特征，2D Tensor，shape为 <code>max(tpWorldSize, 1) * A , H</code>。<br><term>Atlas A2系列产品</term>：不支持共享专家场景。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，2D Tensor，shape为 <code>Bs, K</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>assistInfoForCombine</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>assistInfoForCombineOut</code>输出，1D Tensor，shape为 <code>A * 128,</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>epRecvCounts</code>输出，1D Tensor：<br><term>Atlas A2系列产品</term>：shape为 <code>moeExpertNum + 2 * globalBs * K * serverNum,</code>；<br><term>Atlas A3系列产品</term>：shape为 <code>epWorldSize * max(tpWorldSize, 1) * localExpertNum,</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家权重，2D Tensor，shape为 <code>Bs, K</code>。</td>
   <td>FLOAT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>tpSendCountsOptional</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>tpRecvCounts</code>输出，有TP域通信时传参，否则传空指针：<br><term>Atlas A2系列产品</term>：不支持TP域通信；<br><term>Atlas A3系列产品</term>：有TP域通信时为1D Tensor，shape为 <code>tpWorldSize,</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>输入</td>
   <td>标识token是否参与通信：<br><term>Atlas A2系列产品</term>：预留参数，传空指针；<br><term>Atlas A3系列产品</term>：1D（shape <code>BS,</code>）或2D（shape <code>BS, K</code>）Tensor，BOOL类型，true需排在false前。</td>
   <td>BOOL</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>activationScaleOptional</td>
   <td>输入</td>
   <td>预留参数，当前版本不支持，传空指针。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScaleOptional</td>
   <td>输入</td>
   <td>预留参数，当前版本不支持，传空指针。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>输入</td>
   <td>预留参数，当前版本不支持，传空指针。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOptional</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>expandScales</code>输出：<br><term>Atlas A2系列产品</term>：1D Tensor，shape <code>A,</code>，FLOAT32类型；<br><term>Atlas A3系列产品</term>：预留参数，传空指针。</td>
   <td>FLOAT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>sharedExpertXOptional</td>
   <td>输入</td>
   <td>共享专家计算后的token：<br><term>Atlas A2系列产品</term>：预留参数，传空指针；<br><term>Atlas A3系列产品</term>：2D（shape <code>Bs, H</code>）或3D Tensor（前两维乘积=Bs，第三维=H），数据类型与expandX一致，传入时需满足sharedExpertRankNum=0。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>输入</td>
   <td>RP通信域动态缩容信息：<br><term>Atlas A2系列产品</term>：不支持，传空指针；<br><term>Atlas A3系列产品</term>：1D Tensor（shape <code>4 + 2 * epWoldSize,</code>），INT32类型，前4位为缩容配置，后2*epWoldSize为rank映射表。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>oriXOptional</td>
   <td>输入</td>
   <td>未经过FFN的token数据（公式中的x）：<br><term>Atlas A2系列产品</term>：预留参数，传空指针；<br><term>Atlas A3系列产品</term>：copyExpertNum/constExpertNum>0时必传，2D Tensor（shape <code>Bs, H</code>），数据类型与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>constExpertAlpha1Optional</td>
   <td>输入</td>
   <td>constExpert场景的计算系数（公式中的alpha1）：<br><term>Atlas A2系列产品</term>：预留参数，传空指针；<br><term>Atlas A3系列产品</term>：constExpertNum>0时必传，1D Tensor（shape <code>constExpertNum,</code>），数据类型与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>constExpertAlpha2Optional</td>
   <td>输入</td>
   <td>constExpert场景的计算系数（公式中的alpha2）：<br><term>Atlas A2系列产品</term>：预留参数，传空指针；<br><term>Atlas A3系列产品</term>：constExpertNum>0时必传，1D Tensor（shape <code>constExpertNum,</code>），数据类型与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>constExpertVOptional</td>
   <td>输入</td>
   <td>constExpert场景的计算系数（公式中的v）：<br><term>Atlas A2系列产品</term>：预留参数，传空指针；<br><term>Atlas A3系列产品</term>：constExpertNum>0时必传，2D Tensor（shape <code>constExpertNum, H</code>），数据类型与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>输入</td>
   <td>EP通信域名称（专家并行），字符串长度[1, 128)，不可与groupTp相同。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>输入</td>
   <td>EP通信域大小：<br><term>Atlas A2系列产品</term>：取值16、32、64；<br><term>Atlas A3系列产品</term>：取值[2, 768]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>输入</td>
   <td>EP域本卡ID，取值范围[0, epWorldSize)，同一通信域内不重复。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>输入</td>
   <td>MoE专家数量，满足 <code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code>：<br><term>Atlas A2系列产品</term>：取值(0, 512]，且 <code>moeExpertNum/(epWorldSize - sharedExpertRankNum) ≤24</code>；<br><term>Atlas A3系列产品</term>：取值(0, 1024]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>TP通信域名称（数据并行）：<br><term>Atlas A2系列产品</term>：不支持，传空字符；<br><term>Atlas A3系列产品</term>：字符串长度[1, 128)，不可与groupEp相同。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>TP通信域大小：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, 2]，0/1表示无TP通信，有TP通信时仅支持2。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>输入</td>
   <td>TP域本卡ID：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, 1]，同一通信域内不重复，无TP通信时传0。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>输入</td>
   <td>共享专家卡分布类型：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：仅支持0（共享专家卡排在MoE专家卡前）。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>输入</td>
   <td>共享专家数量：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, 4]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>输入</td>
   <td>共享专家卡数量：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, epWorldSize)，0时需满足sharedExpertNum=0/1，非0时需满足<code>sharedExpertRankNum % sharedExpertNum =0</code>。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>输入</td>
   <td>EP域全局batch size：<br>- 各卡Bs一致时：<code>globalBs = Bs*epWorldSize</code> 或 0；<br>- 各卡Bs不一致时：<code>globalBs = maxBs*epWorldSize</code>（maxBs为单卡Bs最大值）。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>输入</td>
   <td>预留参数，指定输出数据类型，当前版本不支持，传0。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>输入</td>
   <td>通信量化类型（0=不量化，2=int8量化）：<br><term>Atlas A2系列产品</term>：2仅支持commAlg="hierarchy"或特定环境变量配置；<br><term>Atlas A3系列产品</term>：2仅支持<code>tpWorldSize < 2</code>。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>输入</td>
   <td>预留参数，group List格式，当前版本不支持，传0。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>输入</td>
   <td>通信亲和内存布局算法：<br><term>Atlas A2系列产品</term>：支持nullptr、""、"fullmesh"、"hierarchy"，推荐"hierarchy"；<br><term>Atlas A3系列产品</term>：不支持，传空指针。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>输入</td>
   <td>零专家数量：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, MAX_INT32]，专家ID范围<code>[moeExpertNum, moeExpertNum+zeroExpertNum)</code>。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>输入</td>
   <td>copy专家数量：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, MAX_INT32]，专家ID范围<code>[moeExpertNum+zeroExpertNum, moeExpertNum+zeroExpertNum+copyExpertNum)</code>。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>输入</td>
   <td>常量专家数量：<br><term>Atlas A2系列产品</term>：不支持，传0；<br><term>Atlas A3系列产品</term>：取值[0, MAX_INT32]，专家ID范围<code>[moeExpertNum+zeroExpertNum+copyExpertNum, ...)</code>。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>xOut</td>
   <td>输出</td>
   <td>处理后的token，2D Tensor，shape为 <code>Bs, H</code>，数据类型/格式与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输出</td>
   <td>返回Device侧需申请的workspace大小。</td>
   <td>UINT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输出</td>
   <td>返回包含算子计算流程的op执行器。</td>
   <td>aclOpExecutor*</td>
   <td>-</td>
  </tr>
 </tbody>
</table>

### 返回值

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1576px"> 
<colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 400px">
 </colgroup>
 <thead>
  <tr>
   <th>返回值</th>
   <th>错误码</th>
   <th>描述</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>ACLNN_ERR_PARAM_NULLPTR</td>
   <td>161001</td>
   <td>1. 输入和输出的必选参数Tensor是空指针。</td>
  </tr>
  <tr>
   <td>ACLNN_ERR_PARAM_INVALID</td>
   <td>161002</td>
   <td>1. 输入和输出的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
   <td>ACLNN_ERR_INNER_TILING_ERROR</td>
   <td>561002</td>
   <td>1. 输入和输出的shape不在支持的范围内；<br>2. 参数的取值不在支持的范围。</td>
  </tr>
 </tbody>
</table>

## aclnnMoeDistributeCombineV3

### 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"> 
<colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>workspace</td>
   <td>输入</td>
   <td>在Device侧申请的workspace内存地址。</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输入</td>
   <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeDistributeCombineV3GetWorkspaceSize</code>获取。</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输入</td>
   <td>op执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
   <td>stream</td>
   <td>输入</td>
   <td>指定执行任务的Stream。</td>
  </tr>
 </tbody>
</table>

### 返回值

返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- **接口配套约束**：
  - `aclnnMoeDistributeDispatchV3`与`aclnnMoeDistributeCombineV3`必须配套使用，前者输出的`assistInfoForCombineOut`、`epRecvCounts`、`tpRecvCounts`、`expandScales`需直接传入后者对应参数，业务逻辑不可依赖这些Tensor的具体值。

- **参数一致性约束**：
  - 所有卡的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commAlg`参数及`HCCL_BUFFSIZE`取值需保持一致，且与`aclnnMoeDistributeDispatchV3`对应参数一致。
  - 动态缩容后，MOE专家卡的本卡专家数需与缩容前保持一致，无需修改其他参数（缩容信息通过`elasticInfoOptional`传递）。

- **产品特定约束**：
  - <term>Atlas A3系列产品</term>：单卡包含双DIE（晶粒/裸片），参数说明中的“本卡”均指单DIE。
  - 动态缩容功能不支持在TP并行场景下使能。

- **Shape变量约束**：
  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | 本卡需分发的最大token数，需满足产品特定的大小关系（见参数说明），取值与是否缩容相关。 |
  | H（hidden size） | <term>Atlas A2系列产品</term>：(0, 7168]，且为32的整数倍；<br><term>Atlas A3系列产品</term>：[1024, 8192]。 |
  | Bs           | 本卡最终输出token数，<term>Atlas A2系列产品</term>：0 < Bs ≤256；<term>Atlas A3系列产品</term>：0 < Bs ≤512。 |
  | K（topK）    | 0 < K ≤16，且0 < K ≤ <code>moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum</code>。 |
  | serverNum    | 服务器节点数，仅支持2、4、8。                                                  |
  | localExpertNum | 本卡专家数：共享专家卡=1；MoE专家卡=<code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>，>1时不支持TP通信。 |

- **环境变量约束**：
  - **HCCL_BUFFSIZE**（单位MB，默认200MB）：
    - <term>Atlas A2系列产品</term>：根据commAlg取值满足不同大小要求（如"fullmesh"需≥2*(Bs*epWorldSize*min(localExpertNum,K)*H*sizeof(uint16)+2MB)）；
    - <term>Atlas A3系列产品</term>：≥2且≥2*(localExpertNum*maxBs*epWorldSize*Align512(Align32(2*H)+44)+(K+sharedExpertNum)*maxBs*Align512(2*H))（Align512(x)=((x+511)/512)*512，Align32(x)=((x+31)/32)*32）。
  - **HCCL_INTRA_PCIE_ENABLE/HCCL_INTRA_ROCE_ENABLE**：<term>Atlas A2系列产品</term>不再推荐使用，建议通过commAlg="hierarchy"配置。

- **通信域使用约束**：
  - 一个模型中，`aclnnMoeDistributeCombineV3`与`aclnnMoeDistributeDispatchV3`仅支持相同EP通信域，且该域不允许有其他算子。
  - 若使用TP通信域，两者需使用相同TP域或都不使用，TP域不允许有其他算子。

- **其他约束**：
  - 公式中的“/”表示整除。
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>。

## 调用示例


<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> ：类似下文<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>调用示例，其中V3接口相较于V2接口新增的场景参数按上述参数说明传值即可。

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：示例代码如下，仅供参考，调起MoeDistributeCombineV2和MoeDistributeDispatchV2算子（注：接口为V3，算子仍是V2）。




- 文件准备：    
  1.新建combineDemo目录，按照下方指导在combineDemo下新建aclnnCombineDemo.cpp，buildCombine.sh，文件并参考如下代码修改。

  2.安装cann包，并根据下方指导编译运行combineDemo。

-  编译脚本
    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # 更改cann包环境的路径
    g++ "aclnnCombineDemo.cpp" -o combineDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi -lop_common -lpthread -lhccl
    ```
- 编译与运行：

    ```bash
    # source cann环境
    source /path/to/cann_env/latest/bin/setenv.bash

    # 编译aclnnCombineDemo.cpp
    bash buildCombine.sh

    ./combineDemo
    ```

- 示例代码如下，仅供参考
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v3.h"

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

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 8;
    constexpr uint32_t TP_WORLD_SIZE = 2;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

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

    int LaunchOneProcessDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, \
                    context = %p\n", args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream,                 \
                    args.context);

        int64_t Bs = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 1;
        int64_t sharedExpertRankNum = 1;
        int64_t moeExpertNum = 7;
        int64_t quantMode = 0;
        int64_t globalBs = Bs * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        int64_t zeroExpertNum = 1;
        int64_t copyExpertNum = 1;
        int64_t constExpertNum = 1;
        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            A = globalBs / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBs * (localExpertNum < K ? localExpertNum : K);
        }

        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *elasticInfoDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *elasticInfo = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;

        std::vector<int64_t> xShape{Bs, H};
        std::vector<int64_t> expertIdsShape{Bs, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{Bs, K};
        std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE * localExpertNum};
        std::vector<int64_t> expandScalesShape{A};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t elasticInfoShapeSize = GetShapeSize(elasticInfoShape);
        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
        std::vector<float> elasticInfoHostData(elasticInfoShapeSize, 0);
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(elasticInfoHostData, elasticInfoShape, &elasticInfoDeviceAddr, aclDataType::ACL_FLOAT, &elasticInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** 调用dispatch ********************************************/

        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr, 
                expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** 调用combine ********************************************/
        // 调用第一阶段接口
        ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, nullptr, nullptr, nullptr, nullptr,
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, x,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
            return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3                      \
                    execute successfully.\n", args.rankId);

        // 释放device资源
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (elasticInfo != nullptr) {
            aclDestroyTensor(elasticInfo);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (elasticInfoDeviceAddr != nullptr) {
            aclrtFree(elasticInfoDeviceAddr);
        }
        if (elasticInfoDeviceAddr != nullptr) {
            aclrtFree(elasticInfoDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }
        
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
        }

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```