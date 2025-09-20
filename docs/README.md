# 项目文档

## 算子列表
- [算子清单](./context/op_list.md)
- [算子接口（aclnn）](./context/op_api_list.md)

## 算子开发

以`AddExample`算子为例，介绍开发AI Core算子所需的开发交付件和实现过程，请根据实际情况实现对应算子。

- 算子IR（Intermediate Representation）：表示算子原型，描述了算子输入、输出、属性等信息，包括数据类型、shape、数据格式等。算子清单中有部分算子定义了IR，表明可通过IR构图方式调用算子。

| 算子分类 |  算子目录  |    算子IR    |   说明 |
|---------|---------|------------|---------|
|moe|[moe_gating_top_k](../moe/moe_gating_top_k/README.md)|[MoeGatingTopK](../moe/moe_gating_top_k/op_graph/moe_gating_top_k_proto.h)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|
|moe|[moe_re_routing](../moe/moe_re_routing/README.md)|[MoeReRouting](../moe/moe_re_routing/op_graph/moe_re_routing_proto.h)|MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。|
|posembedding|[interleave_rope](../posembedding/interleave_rope/README.md)|[InterleaveRope](../posembedding/interleave_rope/op_graph/interleave_rope_proto.h)|针对单输入 x 进行旋转位置编码。|
|xxx|[xxxx](../xx/xx/graph_plugin/xxx.h)|xxx|xxx|

## 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式也被称为“单算子API调用”，其详细介绍请参见[《AOL算子加速库接口》](https://hiascend.com/document/redirect/CannCommunityOplist)。

本项目提供的所有算子接口清单如下：

|    接口名   |      说明     |
|-----------|------------|
|[aclnnGroupedMatmulSwigluQuant](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|实现融合GroupedMatmul 、dquant、swiglu和quant运算。|
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|实现融合GroupedMatmul 、dquant、swiglu和quant运，是aclnnGroupedMatmulSwigluQuant接口的weightNZ特化版本。|
|[aclnnMoeGatingTopK](../moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|

## 图融合规则

使用图方式描述网络时，可采用图融合提升算子性能。图融合是指[GE（Graph Engine）](https://www.hiascend.com/cann/graph-engine)按融合规则进行改图的过程，使用融合后的算子替换融合前的算子。图融合详细介绍请参见[《图融合和UB融合规则参考》](https://hiascend.com/document/redirect/CannCommunitygraphubfusionref)。

本项目提供的所有融合规则清单如下：

|  规则名  |    说明    |
|---------|------------|
|[MatmulxxxPass](../math/xxx/xx.md)|待补充。|


## 算子开发指南

本章以`AddExample`算子为例，介绍算子具体的开发过程和交付件，开发者请根据实际算子功能自行修改代码实现。
- [AI Core算子开发指南](../docs/context/AI%20Core算子开发指南.md)

## 算子调用

以`AddExample`算子为例，提供如下算子调用方式，请按需选择，详细调用流程参见[算子调用](./context/op_invocation.md)。

- aclnn调用算子 **（推荐）**：以aclnnXxx接口方式调用算子。
- 图模式调用算子：以图方式调用算子。

## 算子调试调优

以`AddExample`算子为例，简单介绍的算子调试、调优方法，请按需选择，使用方法参见[算子调试调优](./context/op_debug_prof.md)。


## 参考资源

开发者学习过程中，可以参考如下资源，了解更多与算子开发、调用、调试调优等相关的知识。

- [基本概念](context/基本概念.md)
- [build参数说明](context/build.md)
- [《CANN 软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)
- [《应用开发（C&C++）》](https://hiascend.com/document/redirect/CannCommunityInferWizard)
- [《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)
- [《算子库接口》](https://hiascend.com/document/redirect/CannCommunityOplist)
- [《Ascend Graph开发指南》](https://hiascend.com/document/redirect/CannCommunityAscendGraph)
- [《图融合和UB融合规则参考》](https://hiascend.com/document/redirect/CannCommunitygraphubfusionref)

