# 算子调用
> 说明：本项目阐述如何与社区版CANN开发套件包配合使用，调用一个简单的AllGather Add通算融合示例算子。
## 前提条件

调用本算子前，请确保已本地下载代码仓，并安装好如下基础依赖、NPU驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - Gawk
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   上述依赖包可通过项目根目录install\_deps.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
   ```bash
   bash install_deps.sh
   ```

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作，安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

## 环境准备

1. **安装社区版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/Ascend-cann-toolkit_8.5.0.alpha001_linux-x86_64.run)、[toolkit aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/Ascend-cann-toolkit_8.5.0.alpha001_linux-aarch64.run)。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. **安装社区版CANN legacy包（运行态依赖）**

    运行算子时必须安装本包，若仅编译算子，可跳过本操作。

    根据产品型号和环境架构，下载对应`cann-${soc_name}-opp_legacy-${cann_version}-linux-${arch}.run`包，下载链接如下：

    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：[legacy x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/cann-910b-ops-legacy_8.5.0.alpha001_linux-86_64.run)、[legacy aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/cann-910b-ops-legacy_8.5.0.alpha001_linux-aarch64.run)。
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品：[legacy x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/cann-910_93-ops-legacy_8.5.0.alpha001_linux-x86_64.run)、[legacy aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/8.5.0.alpha001/cann-910_93-ops-legacy_8.5.0.alpha001_linux-aarch64.run)。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run
    # 安装命令
    ./cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```
    - \$\{soc\_name\}：表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。

    - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **配置环境变量**
    
    根据实际场景，选择合适的命令。

    ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/set_env.sh
   # 指定路径安装
   # source ${install_path}/set_env.sh
    ```
## 编译执行

本示例算子可使用[自定义算子包](#自定义算子包)方式编译执行。

- 自定义算子包：选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。注意自定义算子包优先级高于原始CANN包。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 例如：
    # bash build.sh --pkg --soc=ascend910b --ops=all_gather_add
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子（参见[算子列表](../op_list.md)）。格式形如"apply_rotary_pos_emb,rope_quant_kvcache,..."，多算子之间用英文逗号","分隔。

    说明：若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是built-in包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。
     
    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。
    
2. **安装自定义算子包**
   
    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```
    
    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/latest/opp。注意自定义算子包不支持卸载。

## 本地验证 

通过项目根目录build.sh脚本，可快速调用算子和UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../context/build.md)。

目前本示例算子仅支持API方式（aclnn接口）调用。

- **执行算子样例**

    - 完成自定义算子包安装后，执行命令如下：
        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}]
        # 执行命令示例:
        # bash build.sh --run_example all_gather_add eager cust
        ```

        - \$\{op\}：表示待执行算子，算子名小写下划线形式，如flash_attention_score。
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。         
        - \$\{vendor\_name\}（名称可自定义）：与构建的自定义算子包设置一致，默认名为custom。

        如需执行算子样例，需将自定义算子包安装在默认路径下。执行算子样例后test_aclnn_all_gather_add.cpp文件会按照固定shape随机生成测试数据调用算子，并打印算子与golden的对比执行结果，示例结果如下：
    
        ```
        device_0 aclnnAllGatherAdd execute successfully.
        device_1 aclnnAllGatherAdd execute successfully.
        device_0 aclnnAllGatherAdd golden compare successfully.
        device_1 aclnnAllGatherAdd golden compare successfully.
        ```