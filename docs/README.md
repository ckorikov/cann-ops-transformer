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
本章以XXX算子为例，从如下方面介绍算子的开发过程和对应的交付件：
1. 算子Kernel实现。
2. 算子Tiling实现。
3. xxx

## 算子调用示例
本章以XXX算子为例，分别介绍单算子模式、图模式和主流AI框架（PyTorch）的调用，开发者请根据实际需要选择合适的调用方式。
- [单算子模式调用算子示例](context/单算子调用示例.md)
- [图模式调用算子示例](context/图模式调用示例.md)
- [PyTorch调用算子示例](context/PyTorch调用示例.md)
