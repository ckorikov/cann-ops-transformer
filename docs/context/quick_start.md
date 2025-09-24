# 快速入门
## 目录结构

关键目录如下：

```
├── build.sh                       # 项目工程编译脚本
├── cmake                          # 项目工程编译目录
├── CMakeLists.txt
├── common                         # 项目公共头文件和源码                 
├── docs                           # 项目文档介绍
├── example                        # 项目通用算子开发和调用示例               
├── ffn                            # ffn类算子
├── ...
├── gmm                            # gmm类算子
│   ├── grouped_matmul             # grouped_matmul算子所有交付件，如Tiling、Kernel、InferShape等
│   ├── ...
│   └── CMakeLists.txt             # 算子编译配置文件
├── tests                          # 测试用例目录
├── README.md
├── requirements.txt               # 项目需要的第三方依赖包
└── scripts                        # 脚本目录，包含自定义算子、Kernel构建相关配置文件
```

## 前提条件
> 说明：
> 本项目支持与商发版（8.3.RC1及之前版本）、社区版CANN开发套件包配合使用。商发版CANN包与本项目的配套使用指导请参见“[商发版本说明](./commercial_release.md)”，此处不再赘述。

使用本项目前，请确保如下基础依赖、NPU驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

2. **安装驱动与固件（可选）**

   如需本地运行项目算子，请参见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》，按要求完成NPU驱动和固件软件包安装；否则可跳过本操作。

## 环境准备

1. **安装社区版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/2025091701_newest/Ascend-cann-toolkit_8.3.RC1_linux-x86_64_tmp.run)、[aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/2025091701_newest/Ascend-cann-toolkit_8.3.RC1_linux-aarch64_temp.run)。
    
    安装命令如下：

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，toolkit包将安装在\$\{install\_path\}/ascend-toolkit目录下。

2. **安装社区版CANN legacy包（可选）**

    如需本地运行项目算子，需额外安装此包，否则跳过本操作。

    根据产品型号和环境架构，下载对应`${soc_version}-opp_legacy-${cann_version}-linux.${arch}.run`包，下载链接如下：

    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：[x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20250917_newest/Ascend910B-opp_legacy-8.3.t12.0.b087-linux.x86_64.run)、[aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20250917_newest/Ascend910B-opp_legacy-8.3.t12.0.b087-linux.aarch64.run)

    安装命令如下：

    ```bash
    # 确保安装包具有可执行权限
    chmod +x ${soc_version}-opp_legacy-${cann_version}-linux.${arch}.run
    # 安装命令
    ./${soc_version}-opp_legacy-${cann_version}-linux.${arch}.run --full --install-path=${install_path}/ascend-toolkit
    ```
    - \$\{soc\_version\}：表示NPU型号。
    - \$\{install\_path\}：表示指定安装路径，需要要toolkit包安装在相同路径。

3. **配置环境变量**
	
	根据实际场景，选择合适的命令。

    ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 指定路径安装
   # source ${install-path}/ascend-toolkit/set_env.sh
    ```

4. **下载源码**

    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/ops-transformer-dev.git
    # 安装根目录requirements.txt依赖
    pip3 install -r requirements.txt
    ```

## 编译执行

> 说明：
>
> 基于商发版（8.3.RC1及之前版本）CANN开发套件包对算子源码定制化修改时，请使用“自定义算子包”方式编译和安装。

基于社区版CANN开发套件包对AI Core算子源码定制化修改时，可使用[ops-transformer包](#ops-transformer包)和[自定义算子包](#自定义算子包)方式编译和安装；若对AI CPU算子源码定制化修改时，请使用[自定义算子包](#自定义算子包)方式编译和安装。

编译方式说明：

- ops-transformer包：选择整个项目编译生成的包称为ops-transformer包，可**完整替换**CANN软件包对应部分。
- 自定义算子包：选择项目部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN软件包，不改变其原始包内容。注意自定义算子包优先级高于原始CANN软件包。

### ops-transformer包

1. **编译ops-transformer包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：推荐设置，表示不编译算子的二进制文件。
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。

    若提示如下信息，则说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-transformer_${cann_version}_linux.${arch}.run" successfully created.
    ```

    \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。
    

2. **安装ops-transformer包**
   
    ```bash
    ./cann-${soc_name}-ops-transformer_${cann_version}_linux.${arch}.run --full --install-path=${install_path}/ascend-toolkit
    ```
    
    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名称，不指定时默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，全量算子参见[算子列表](./op_list.md)，不指定时默认编译所有算子。格式形如"abs,add_lora,..."，多算子之间用英文逗号","分隔。
    
    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**
   
    ```bash
    ./cann-ops-math-${vendor_name}_linux-${arch}.run
    ```
    
    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}表示CANN toolkit包安装路径，一般为\$\{install\_path\}/ascend-toolkit/latest，可通过环境变量配置。 

## 本地验证

通过根目录的build.sh脚本执行算子样例、UT用例等，build参数介绍参见[build参数说明](./build.md#参数说明)。

- **执行算子样例**
  
    - 完成ops-transformer包安装后，可执行项目中已有算子的调用样例。编译执行命令如下：

        ```bash
        bash build.sh --run_example ${op} ${mode}
        ```

        - \$\{op\}：表示待执行样例的算子名。
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。

        执行完成后会打印运行结果。

    - 完成ops-transformer-${vendor_name}包安装后，可执行项目中已有算子的调用样例。编译执行命令如下：
    
        ```bash
        # 方式1：不指定vendor_name，默认名为custom
        bash build.sh --run_example ${op} ${mode} ${pkg_mode}
        # 方式2：指定vendor_name
        # bash build.sh --run_example ${op} ${mode} ${pkg_mode} --vendor_name=${vendor_name}
        ```

        - \$\{op\}：表示待执行样例的算子名。
        - \$\{mode\}：表示执行模式，目前仅支持eager（aclnn调用）、graph（图模式调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
        - \$\{vendor\_name\}：表示构建的自定义算子包名称。

        执行完成后会打印运行结果。
        以Abs算子的运行结果为例，运行后的结果示例如下：

        ```
        mean result[0] is: 1.000000
        mean result[1] is: 1.000000
        mean result[2] is: 1.000000
        mean result[3] is: 2.000000
        mean result[4] is: 2.000000
        mean result[5] is: 2.000000
        mean result[6] is: 3.000000
        mean result[7] is: 3.000000
        ```
  
- **执行算子UT**

   安装完编译生成的包后，可执行UT验证功能是否正常，具体命令如下：

   > 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

   ```bash
   # 方式1: 编译并执行所有的UT测试用例
   bash build.sh -u
   # 方式2: 编译所有的UT测试用例但不执行
   # bash build.sh -u --noexec
   # 方式3: 编译并执行对应功能的UT测试用例（选其一）
   # bash build.sh -u --[opapi|ophost|opkernel]
   # 方式4: 编译对应功能的UT测试用例但不执行（选其一）
   # bash build.sh -u --noexec --[opapi|ophost|opkernel]
   ```

   假设验证ophost功能是否正常，执行如下命令：
   ```bash
   bash bulid.sh -u --ophost
   ```

   执行完成后出现如下内容，表示执行成功。
   ```bash
   Global Environment TearDown
   [==========] ${n} tests from ${m} test suites ran. (${x} ms total)
   [  PASSED  ] ${n} tests.
   [100%] Built target transformer_op_host_ut
   ```
   其中\$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。