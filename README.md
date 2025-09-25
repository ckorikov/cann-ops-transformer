# ops-transformer

## 🔥Latest News

- [2025/09] ops-transformer项目首次上线。

## 🚀概述

ops-transformer是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供transformer类大模型计算的进阶算子库，包括attention类、moe类等算子，算子库架构图如下：

<img src="docs/figures/architecture.png" alt="架构图"  width="750px" height="400px">

## ⚡️快速入门

若您希望快速体验算子的调用和开发过程，请访问如下文档获取简易教程。

- [算子调用](docs/context/quick_op_invocation.md)：介绍调用算子的基本步骤，快速搭建环境，实现算子编译执行。
- [算子开发](docs/context/quick_op_develop.md)：介绍开发算子的基本流程，一键创建算子工程目录，实现Tiling、Kernel核心交付件。

## 📖学习教程

若您希望深入体验项目功能并修改算子源码，请访问如下文档获取详细教程。
- [算子调用方式](docs/context/op_invocation.md)：介绍不同的调用算子方式，方便快速应用于不同的AI业务场景。

- [算子调试调优](docs/context/op_debug_prof.md)：介绍常见的算子调试和调优方法，如DumpTensor、msProf等。

- [算子基本概念](docs/context/基本概念.md)：介绍算子领域相关术语和概念，如非连续Tensor、量化模式等。

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
