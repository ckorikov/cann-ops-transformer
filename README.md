# ops-transformer

## Latest News🔥

- [2025/09] ops-transformer项目首次上线。

## 🚀概述

ops-transformer是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供transformer类大模型计算的算子库，包括gmm类、moe类等，全量算子清单请参见[算子清单](docs/context/op_list.md)。

![原理图](docs/figures/architecture.png)


## ⚡️快速入门

若您希望快速体验项目，请访问[快速入门](docs/context/quick_start.md)获取简易教程，包括目录结构介绍、环境搭建、编译执行、本地验证等操作。

- [目录结构](docs/context/quick_start.md#目录结构)：体验项目之前，请先了解项目包含的关键目录结构和对应交付件的含义。
- [前提条件](docs/context/quick_start.md#前提条件)：安装软件包之前，请完成基础环境搭建，包括第三方依赖、NPU驱动和固件等。
- [环境准备](docs/context/quick_start.md#环境准备)：基础环境搭建后，需完成社区版CANN软件包安装、环境变量配置、源码下载等。
- [编译执行](docs/context/quick_start.md#编译执行)：环境准备好后，可对算子源码修改（如优化、新增等），编译生成的算子包可部署到AI业务中。
- [本地验证](docs/context/quick_start.md#本地验证)：基于项目根目录的build.sh脚本，可执行算子样例、UT用例等，快速验证项目功能。

## 📖学习教程

若您希望深入体验项目功能并定制化修改算子源码，请访问[项目文档](./docs/README.md)获取相关文档，如算子清单和接口、算子开发指南、算子调用和算子调试调优方法、build参数说明等。

- [算子开发](docs/README.md#算子开发)：介绍算子端到端开发流程，包括算子规格设计、原型定义、Tiling策略、Kernel实现、框架适配等。
- [算子调用](docs/README.md#算子调用)：介绍不同调用算子的方式，方便快速应用于实际AI业务中。
- [算子调试调优](docs/README.md#算子调试调优)：介绍常见的算子调试和调优方法，如DumpTensor、msProf等。

## 📝相关信息
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
