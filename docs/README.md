# 项目文档

## 产品文档

- [AOL算子加速库接口](https://hiascend.com/document/redirect/CannCommercialOplist)
- [图融合和UB融合规则参考](https://hiascend.com/document/redirect/CannCommercial-graphubfusionref)
- [应用开发（C&C++）> 单算子调用](https://hiascend.com/document/redirect/canncommercial-aclcppdevg)

## 基本概念
-   [两段式接口](./context/两段式接口.md)
-   [数据结构](./context/数据结构.md)
-   [数据类型](./context/数据类型.md)
-   [数据格式](./context/数据格式.md)
-   [非连续的Tensor](./context/非连续的Tensor.md)
-   [broadcast关系](./context/broadcast关系.md)
-   [互推导关系](./context/互推导关系.md)
-   [互转换关系](./context/互转换关系.md)
-   [量化介绍](./context/量化介绍.md)
-   [sparse模式介绍](./context/sparse_mode参数说明.md)

## 算子原型
| 算子分类 |  算子名  |    说明    |
|---------|---------|------------|
|xxx|[xxxx](../xx/xx/graph_plugin/xxx.h)|xxx|

## 算子接口（aclnn）
|    接口名   |      说明     |
|-----------|------------|
|[aclnnGroupedMatmulSwigluQuant](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|实现融合GroupedMatmul 、dquant、swiglu和quant运算。|
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|实现融合GroupedMatmul 、dquant、swiglu和quant运，是aclnnGroupedMatmulSwigluQuant接口的weightNZ特化版本。|

## 融合规则
|  融合Pass  |    说明    |
|---------|------------|
|[xxPass](../math/xxx/xx.md)|xxxx。|

## 算子开发示例


## 算子开发指南

本章以`AddExample`算子为例，介绍算子具体的开发过程和交付件，开发者请根据实际算子功能自行修改代码实现，具体过程请参见[算子开发样例](../example/add_example/README.md)。

## 算子调用

本章提供了常见的算子调用方式：

- aclnn调用算子 **（推荐）**：以aclnnXxx接口方式调用算子。
- 图模式调用算子：以IR构图方式调用算子。

开发者按需选择，不同方式下算子的调用流程和关键代码实现请参见[算子调用样例](./context/算子调用样例.md)。
