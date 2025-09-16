# ops-transformer

## Latest News🔥

| 最新动态            | 更新时间   |
|-----------------| ---------- |
| ops-transformer项目首次上线。 | 2025-09-30 |

## 概述

本项目是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供transformer类大模型算子的高阶算子库，简称ops-transformer，涵盖了常见的FlashAttention、MoE（Mixture of Experts）等算子。

ops-transformer在CANN架构中的位置如下图所示：

![原理图](docs/figures/architecture.png)

算子库提供了丰富的深度优化、硬件亲和的高性能算子，为AI网络在昇腾硬件上加速计算奠定基础。这些算子分为高阶算子、基础算子和算子框架层，自上而下，上层算子运行依赖下层算子。具体来说，高阶算子依赖基础算子，而所有算子都依赖底层框架。

算子库按功能可划分为：

- [ops-nn](https://gitcode.com/cann/ops-nn-dev)：指神经网络（Neural Network，NN）类算子仓，提供诸如Matmul等算子。
- [ops-cv](https://gitcode.com/cann/ops-cv-dev)：指计算机视觉（Computer Vision）类算子仓，提供诸如GridSample等算子。
- **ops-transformer（本项目）**：指transformer类大模型算子仓，提供诸如FlashAttention、MoE（Mixture of Experts）等算子。
- [ops-math](https://gitcode.com/cann/ops-math-dev)：指数学类基础算子仓，提供诸如Add、Abs等算子。
- [ops-base](https://gitcode.com/cann/ops-base-dev)：指算子基础框架仓，提供基础的调度能力（如Tensor创建/释放、workspace复用等）和公共依赖项（如公共头文件、公共结构体、公共调度框架等）。


希望开发者通过学习本项目能掌握aclnn接口、图模式以及主流AI框架（如PyTorch等）调用算子的方法，深入了解算子的开发过程，能在实际业务中开发高性能算子。

## 版本配套说明

  - 本项目会创建与CANN软件版本适配的标签并发行，两者的配套关系请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"。**需注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与GitCode标签源码，使用master分支可能存在版本不匹配风险。**

  - 本项目支持的固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=28)”，根据产品型号与CANN软件版本获取配套的固件与驱动。


## 目录结构说明

ops-transformer项目关键目录如下：

```
├── build.sh                       # 项目工程编译脚本
├── cmake                          # 项目工程编译目录
├── CMakeLists.txt
├── common
│   ├── include                    # 项目公共头文件
│   ├── src                        # 项目公共接口实现源代码
│   └── ...
├── docs                           # 项目内算子使用说明和文档介绍
├── example                        # 使用示例:算子分类目录
│   ├── add_example                # 示例：算子名
│   │   ├── CMakeLists.txt         # 算子编译配置文件，保留原文件即可   
│   │   ├── examples               # 算子使用示例
│   │   ├── op_graph               # 算子构图相关目录
│   │   ├── op_host                # 算子信息库、Tiling、InferShape相关实现目录
│   │   └── op_kernel              # 算子Kernel目录
│   ├── README.md                  # 算子说明文档
│   └── CMakeLists.txt             # 算子编译配置文件，保留原文件即可
├── LICENSE
├── ...
├── gmm                            # gmm类算子所有交付件，包括算子实现、构图等
│   ├── grouped_matmul_swiglu_quant# gmm类算子中的grouped_matmul_swiglu_quant算子
│   │   ├── docs                   # 算子说明文档
│   │   ├── example                # 算子使用示例
│   │   ├── op_graph               # 算子构图相关目录
│   │   ├── op_host                # 算子信息库、Tiling、InferShape相关实现目录
│   │   │   └── op_api             # 算子aclnn接口实现目录
│   │   ├── tests                  # 测试用例目录
│   │   ├── README.md              # 算子说明文档
│   │   └── ...
│   └── ...
├── mc2                            # mc2类算子所有交付件，包括算子实现、构图等
│   ├── 3rd                        # mc2类算子中的3rd算子
│   │   ├── batch_mat_mul_v3       # 3rd算子中batch_mat_mul_v3算子
│   │   │   ├── op_host            # 算子host目录
│   │   │   └── op_kernel          # 算子Kernel目录
│   │   └── ...
│   └── ...
├── README.md
├── requirements.txt               # 本项目需要的第三方依赖包
└── scripts                        # 脚本目录，包含自定义算子、Kernel构建相关配置文件
```


## 环境准备
> 说明：
>
> 本项目支持与CANN 8.3.RC1及之前商发版本的开发套件`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`配合使用，
> 使用指导请参见CANN 8.3.RC1商发文档中“[版本说明](https://idp.huawei.com/idp-designer-war/design?op=edit&locate=newMode/EDIT/53011869427/zh-cn_BOOKMAP_0000002456008653/ZH-CN_TOPIC_0000002422451490/2)”。

ops-transformer项目支持源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **获取软件包**

   请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"获取对应的CANN软件包`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`。
   - \$\{package\}表示待安装的CANN软件包名
   - \$\{cann\_version\}表示CANN包版本号
   - \$\{arch\}表示CPU架构，如aarch64、x86_64
   
   为确保您的源码定制开发顺利，请选择与CANN版本配套的GitCode分支源码，使用master分支可能存在版本不匹配风险。

2. **安装软件包**

   注意，执行安装命令时，请确保安装用户对软件包具有可执行权限。

   基础环境的搭建请参见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》，按要求完成NPU驱动和固件、`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`软件包的安装。

3. **安装依赖**
   ops-transformer源码编译用到的依赖如下，请确保已安装并且满足版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.8）
   - dos2unix
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   - 本项目需要使用的python依赖包，具体参见工程目录中requirements.txt，可通过如下命令一键安装：
     ```bash
     pip3 install -r requirements.txt
     ```

4. **环境变量配置**

    根据实际场景，选择合适的命令。

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 指定路径安装
    source ${install-path}/ascend-toolkit/set_env.sh
    ```

## 源码下载
开发者可以通过如下命令下载本仓源码：

  ```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-transformer-dev.git
  ```

\$\{tag\_version\}请替换为具体的标签名称，本项目与CANN版本配套关系请参见[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)。

## 编译执行
> 说明：
>
> 若基于CANN 8.3.RC1及之前商发版本的开发套件包`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`进行算子源码定制化修改，请使用“自定义算子包”方式进行编译和安装。

基于CANN软件包`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`进行算子源码定制化修改时，支持使用[自定义算子包](#自定义算子包)和[ops-transformer包](#ops-transformer包)方式进行编译和安装。

编译方式说明：

- 自定义算子包：选择ops-transformer项目中部分算子编译生成的包称为自定义算子包，不改变原始CANN软件包，**挂载优先级更高的自定义算子包**。
- ops-transformer包：选择ops-transformer完整项目编译生成的包称为ops-transformer包，可**完整替换**CANN软件包中对应的部分。

### 自定义算子包

1. **编译自定义算子包。**

    进入本项目根目录，执行如下编译命令：
    
    ```bash
    # 方式1：编译所有算子
    bash build.sh
    # 方式2：编译指定算子，如op1、op2
    bash build.sh -n "op1;op2"
    ```
    - -n（可选）：**仅编译部分算子设置**。"op1;op2"表示待编译的算子，多个算子之间使用英文分号";"分隔并使用引号。

2. **安装自定义算子包。**
  安装前，需确保所安装的自定义算子包与所安装CANN开发套件包CPU架构一致，并且要先设置CANN开发套件包环境变量，然后再进行安装，仅支持在配套版本安装自定义算子包，安装命令如下：
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh # 设置CANN开发套件包环境变量，以root用户默认路径为例，如已设置，则请忽略该操作
    ./CANN-custom_ops-<cann_version>-linux-${arch}.run
    ```

    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，其中\$\{ASCEND\_HOME\_PATH\}已在[环境准备](#环境准备)章节通过环境变量配置操作完成。

### ops-transformer包

1. **编译ops-transformer包。**
    进入本项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --build package
    ```
    若提示如下信息，则说明编译成功。

    ```bash
    Self-extractable archive "CANN-ops-transformer-${cann_version}-linux.${arch}.run" successfully created.
    ```

    编译成功后，run包存放于build_out目录下。

2. **安装ops-transformer包。**
   
    ```bash
    ./CANN-ops-transformer-${cann_version}-linux.${arch}.run --full --quiet 
    ```

    ops-transformer包默认安装路径为：`/usr/local/Ascend`，如需自定义安装路径，可使用"--install-path"参数指定。

## 本地验证 
正在适配


## 相关文档

为方便开发者学习本项目，请先访问[docs目录](docs/README.md)，按需获取对应文档，文档内容包括：

| 文档                                                  | 说明                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| [基本概念](docs/README.md#基本概念)                   | 介绍项目相关的基础概念和特性，例如术语（包括量化、稀疏等）、算子数据类型或数据格式等。 |
| [算子清单](docs/README.md#算子清单)                   | 介绍项目包含的所有算子清单。                                 |
| [算子接口（aclnn）](docs/README.md#算子接口（aclnn）) | 介绍项目包含的所有aclnn前缀的算子API，通过该API可直调算子。  |
| [图融合规则](docs/README.md#图融合规则)               | 介绍项目支持的所有图融合规则。                               |
| [算子开发指南](docs/README.md#算子开发指南)           | 介绍实现本项目算子的开发流程，包括算子原型定义、Kernel实现、Tiling策略、接口开发等。 |
| [算子调用](docs/README.md#算子调用)                   | 介绍不同场景下调用算子的方式，方便您快速在AI业务中应用。     |
| [算子调试调优](docs/README.md#算子调试调优)           | 介绍算子调试、调优的方法。                                   |
| [参考文档](docs/README.md#参考文档)                   | 介绍项目相关的产品手册，例如《应用开发指南》、《算子加速库接口参考》等。 |
| [附录](docs/README.md#附录)                           | 介绍项目中关键的工程脚本或文件、FAQ等，如**build.sh参数说明**。 |


## 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

- 算子Bug修复

  如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

- 算子优化

  如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行跟踪优化。

- 贡献新算子

  如果您有全新的算子想基于NPU进行设计实现，欢迎您在Issue中提出新的想法和设计。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue提供新增算子说明和设计方案，项目成员会与您进行沟通确认，并为您的算子提供一个合适的`contrib`目录分类，您可以将新增算子贡献到对应目录下。

  同时，您需要在提交的Issue中评论“/assign”或“/assign @yourself”，认领该Issue并在后续完成新算子上库。

  新增算子的交付件通常比较多，您可以参考如下列表检查最小交付件集合，其中`${op_name}`表示新增算子名称：
  ```
  ${op_class}                                          # 算子分类
  ├── ${op_name}                                       # 算子名
  │   ├── op_host                                      # 算子信息库、Tiling、InferShape相关实现
  │   │   ├── ${op_name}_def.cpp                       # 算子信息库定义文件
  │   │   ├── ${op_name}_tiling.cpp                    # 算子Tiling实现文件
  │   │   └── CMakeLists.txt
  │   ├── op_kernel                                    # 算子Kernel目录
  │   │   ├── ${op_name}.cpp
  │   │   ├── ${op_name}.h
  │   │   ├── ${op_name}_tiling_data.h
  │   │   └── ${op_name}_tiling_key.h
  │   ├── CMakeLists.txt                               # 算子编译配置文件，保留原文件即可
  │   └── README.md                                    # 算子说明文档
  ```

- 文档纠错

  如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。

- 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。

## 安全声明
[ops-transformer安全声明](SECURITY.md)

## 许可证
[CANN Open Software License Agreement Version 1.0](LICENSE)