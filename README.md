# ops-transformer


## 概述

本项目是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供transformer类大模型算子的高阶算子库，简称ops-transformer，涵盖了常见的FlashAttention、MoE（Mixture of Experts）等算子。

ops-transformer在CANN架构中的位置如下图所示：

![原理图](docs/figures/architecture.png)

算子库提供了丰富的深度优化、硬件亲和的高性能算子，为AI网络在昇腾硬件上加速计算奠定基础。这些算子分为高阶算子、基础算子和算子框架层，自上而下，上层算子运行依赖下层算子。具体来说，高阶算子依赖基础算子，而所有算子都依赖底层框架。

算子库按功能可划分为：

- ops-nn：指神经网络（Neural Network，NN）类算子仓，提供诸如Matmul等算子。
- ops-cv：指计算机视觉（Computer Vision）类算子仓，提供诸如GridSample等算子。
- **ops-transformer（本项目）**：指transformer类大模型算子仓，提供诸如FlashAttention、MoE（Mixture of Experts）等算子。
- ops-math：指数学类基础算子仓，提供诸如Add、Abs等算子。
- ops-base：指算子基础框架仓，提供基础的调度能力（如Tensor创建/释放、workspace复用等）和公共依赖项（如公共头文件、公共结构体、公共调度框架等）。


通过学习本项目，希望开发者不仅能掌握单算子模式、图模式以及主流AI框架（如PyTorch等）调用算子的方法，还能深入了解算子的开发过程，能在实际业务中开发高性能算子，从而实现模型在NPU上的加速计算。

## 版本配套说明

  - 本项目会创建与CANN软件版本适配的标签并发行，两者的配套关系请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"。**需注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与GitCode标签源码，使用master分支可能存在版本不匹配风险。**

  - 本项目支持的固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=28)”，根据产品型号与CANN软件版本获取配套的固件与驱动。


## 目录结构说明

ops-transformer项目关键目录如下：
```
├── docs                           # 项目公共文档介绍
├── common                             
|   ├── inc                        # 项目公共头文件
│   ├── src                        # 项目公共接口实现源代码
├── gmm                            # GroupedMatmul类算子所有交付件，包括算子实现、接口调用、融合规则等
├── ......
├── examples                       # 项目工程使用示例
├── cmake   
├── CMakeLists.txt
├── tools                          # 项目公共的工具脚本，如测试框架代码
├── LICENSE
├── README.md                      
└── build.sh                       # 项目工程编译脚本
```

## 环境准备
ops-transformer项目支持源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **获取软件包**

   请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"获取对应的CANN开发套件包`Ascend-cann-toolkit_<cann_version>_linux-<arch>.run`、算子二进制包`Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run`（二进制包，算子运行时依赖）、math类基础算子包`CANN-opp-math-<cann_version>-linux.<arch>.run`和算子基础框架包`CANN-ops-base-<cann_version>-linux.<arch>.run`。

   - 为确保您的源码定制开发顺利进行，请选择配套的CANN版本与GitCode分支源码，使用master分支可能存在版本不匹配的风险。
   - 操作系统软件包获取和安装请参见[用户手册](https://hiascend.com/document/redirect/CannCommunityInstSoftware)。

2. **安装驱动和依赖**

   ops-transformer项目运行依赖昇腾NPU驱动和固件，其安装过程参见[用户手册](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，先选择安装场景，再按“准备软件包”、“准备用户”、“安装NPU驱动和固件”章节完成驱动和固件安装。

   此外，ops-transformer源码编译用到的依赖如下，其中python、gcc安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，先选择安装场景，再按“安装CANN > 安装依赖”章节完成相关依赖的安装。

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - protobuf <=3.20.x

     算子编译时，protobuf版本需低于3.20.x，您可以执行**pip3 list**命令查询当前环境中的protobuf版本，如果版本高于3.20.x，则执行如下命令重新安装，以重新安装3.20.0版本为例：

     ```bash
     pip3 install protobuf==3.20.0
     ```

     如果使用非root用户安装，需要在安装命令后加上--user，例如**pip3 install protobuf==3.20.0 --user**。

   - googletest（可选，仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

     如下以[googletest源码](https://github.com/google/googletest.git)编译安装为例，安装命令如下：

     ```bash
     mkdir temp && cd temp                 # 在googletest源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

3. **安装软件包**

   执行安装命令时，请确保安装用户对软件包具有可执行权限。

   - 使用默认路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install
     # 算子二进制包安装命令示例：
     ./Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run --install
     # 算子基础框架包安装命令示例：
     ./CANN-ops-base-<cann_version>-linux.<arch>.run --full --quite
     # math类基础算子包安装命令示例：
     ./CANN-opp-math-<cann_version>-linux.<arch>.run --full --quite
     ```

     - 若使用root用户安装，安装完成后CANN开发套件包存储在`/usr/local/Ascend/ascend-toolkit/latest`路径；算子二进制包存储在`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径；算子基础框架包存储在`/usr/local/Ascend/latest/ops-base`路径；math类基础算子包存储在`/usr/local/Ascend/<cann_version>/opp/built-in/op_impl/ai_core/tbe/impl/ascendc`路径。
     - 若使用非root用户安装，安装完成后CANN开发套件包存储在`$HOME/Ascend/ascend-toolkit/latest`路径；算子二进制包存储在`${HOME}/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径；算子基础框架包存储在`$HOME/Ascend/latest/ops-base`路径；math类基础算子包存储在`$HOME/Ascend/<cann_version>/opp/built-in/op_impl/ai_core/tbe/impl/ascendc`路径。

   - 指定路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --install-path=${install_path}
     # 算子二进制包安装命令示例：
     ./Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run --install --install-path=${install_path}
     # 算子基础框架包安装命令示例：
     ./CANN-ops-base-<cann_version>-linux.<arch>.run --full --quite --install-path=${install_path}
     # math类基础算子包安装命令示例：
     ./CANN-opp-math-<cann_version>-linux.<arch>.run --full --quite --install-path=${install_path}
     ```

     安装完成后，CANN开发套件包存储在`${install_path}/ascend-toolkit/latest`指定路径；算子二进制包存储在`${install_path}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径；算子基础框架包存储在`${install_path}/Ascend/latest/ops-base`路径；math类基础算子包存储在`${install_path}/Ascend/<cann_version>/opp/built-in/op_impl/ai_core/tbe/impl/ascendc`路径。

4. **设置环境变量**

   - 默认路径，root用户安装

     ```bash
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     ```

   - 默认路径，非root用户安装

     ```bash
     source $HOME/Ascend/ascend-toolkit/set_env.sh
     ```

   - 指定路径安装

     ```bash
     source ${install_path}/ascend-toolkit/set_env.sh
     ```

   **注意：若环境中已安装多个版本的CANN软件包，设置上述环境变量时，请确保${install_path}/ascend-toolkit/latest目录指向的是配套版本的软件包。**


## 源码下载
开发者可以通过如下命令下载本仓源码：

  ```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-transformer-dev.git
  ```

${tag_version}请替换为具体的标签名称，本源码仓与CANN版本的配套关系可参见[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)。


## 编译执行
### 自定义算子包编译

进入本仓代码根目录，执行如下命令：

  ```bash
  mkdir build && cd build     # 在融合算子源码根目录下创建临时目录并进入
  cmake ..
  make package -j 并发数      # 编译并生成自定义算子run包，并发数请替换为实际取值
  ```

**说明：**

编译时间较长，请耐心等待：在无缓存场景，使用72核编译器，**-j 144**并发执行，编译大约耗时13分钟。您可以通过**grep 'processor' /proc/cpuinfo | wc -l**命令查询当前服务器cpu核数，并发数=cpu核数*2。

若提示如下信息，则说明编译成功。

  ```
  Self-extractable archive "CANN-custom_ops-<cann_version>-linux.<arch>.run" successfully created.
  ```

编译成功后在 `本仓代码根目录/output` 目录生成自定义算子包：`CANN-custom_ops-<cann_version>-linux.<arch>.run`。

其中，\<cann_version>表示软件版本号，\<arch>表示操作系统架构。

### 自定义算子包安装<a name="2"></a>

安装前，需确保所安装的自定义算子包与所安装CANN开发套件包CPU架构一致，并且要先设置CANN开发套件包环境变量，然后再进行安装，仅支持在配套版本安装自定义算子包，安装命令如下：

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh # 设置CANN开发套件包环境变量，以root用户默认路径为例，如已设置，则请忽略该操作
  ./CANN-custom_ops-<cann_version>-linux.<arch>.run --quiet         # 安装自定义算子run包
  ```

执行上述命令后，自定义算子run包会默认安装到CANN软件包目录，例如，`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/` 目录。

### 单元测试编译执行

UT（单元测试用例），用来看护编译是否正常，进入本仓代码根目录，依次执行如下命令：

  ```bash
  mkdir build && cd build             # 在融合算子源码根目录下创建临时目录并进入
  cmake .. -DTESTS_UT_OPS_TEST=ALL    # 指定编译所有融合算子的单元测试用例
  make ops_test_utest -j 并发数        # 编译并执行所有融合算子的单元测试用例，并发数请替换为实际取值
  ```

执行UT用例依赖googletest单元测试框架，关于googletest更多功能请参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

### 示例工程编译执行

此操作需要在真实NPU环境上进行，并且依赖CANN开发套件包和算子二进制包，因此在编译前，需要参见[环境准备](#1)章节安装配套版本的CANN开发套件包和算子二进制包，并设置环境变量，然后进入本仓代码根目录，依次执行如下命令：

  ```bash
  mkdir build && cd build                 # 在融合算子源码根目录下创建临时目录并进入
  cmake .. -DTESTS_EXAMPLE_OPS_TEST=ALL   # 指定编译所有examples用例示例
  make                                    # 编译并执行所有examples用例
  ```

上述cmake编译参数详细解释请参见[cmake编译参数说明](./docs/common/cmake编译参数说明.md)。

**说明**：当前还提供了一键式编译脚本，进入本仓代码根目录，可执行命令如下，您还可以通过**bash build.sh --help**命令查询更多可用参数；执行完自定义算子包一键式编译命令，需要完成[自定义算子包安装](#2)后，才能执行后续命令。

  ```bash
  bash build.sh         #自定义算子包编译
  bash build.sh -t      #单元测试编译执行
  bash build.sh -e      #示例工程编译执行
  ```


## 回滚
待补充


## 相关文档

为方便开发者学习本项目，请先访问[docs目录](docs/README.md)，按需获取对应文档，文档内容包括：

| 内容                                         | 说明                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| [产品文档](docs/README.md#产品文档)          | 介绍项目相关的所有产品手册，包括《应用开发指南》、《算子加速库接口参考》等。 |
| [基本概念](docs/README.md#基本概念)          | 介绍项目相关的基础概念和特性，例如术语（包括量化、稀疏等）、算子数据类型或数据格式等。 |
| [算子原型](docs/README.md#算子原型)          | 介绍项目包含的所有算子原型。                                 |
| [算子接口](docs/README.md#算子接口（aclnn）) | 介绍项目包含的所有算子API，即以aclnn为前缀的API。            |
| [融合规则](docs/README.md#融合规则)          | 介绍项目包含的所有算子融合规则。                             |
| [算子开发示例](docs/README.md#算子开发示例)  | 介绍实现本项目算子的开发流程，包括算子原型定义、Kernel实现、Tiling策略、接口开发等。 |
| [算子调用示例](docs/README.md#算子调用示例)  | 介绍不同场景下调用本项目算子的方法，方便您快速在AI业务中应用。 |

## 贡献指南

ops-transformer项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，以及参与源码仓贡献的详细流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。


## 许可证
[CANN Open Software License Agreement Version 1.0](LICENSE)