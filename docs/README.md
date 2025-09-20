# 项目文档

## 算子列表
- [算子清单](./context/op_list.md)
- [算子接口（aclnn）](./context/op_api_list.md)

## 算子开发

以`AddExample`算子为例，介绍开发AI Core算子所需的开发交付件和实现过程，请根据实际情况实现对应算子。

- [AI Core算子开发指南](../docs/context/aicore_develop_guide.md)

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

