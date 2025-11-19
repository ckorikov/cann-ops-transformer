# 算子调用
> 说明：本项目阐述如何与社区版CANN开发套件包配合使用，调用一个简单的AllGather Add通算融合示例算子。
## 前置准备

调用本算子前，请确保已本地下载代码仓，并安装好如下基础依赖、NPU驱动和固件已安装。

本项目源码编译用到的依赖如下，请参考[算子调用](https://gitcode.com/cann/ops-transformer/blob/e9d9680f438bfc9fa5587a8a7d4e4b543e9e90bf/docs/invocation/quick_op_invocation.md)文档中的**前提条件**和**环境准备**章节，完成环境依赖的下载和CANN包的准备，其中，**环境准备**章节中的**安装社区版CANN ops-math包**部分可以跳过。

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