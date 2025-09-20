# 商发版本说明

若您安装了**商发版**（**8.3.RC1及之前**）CANN开发套件包`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`，需要对部分算子源码进行定制化修改，可通过修改对应开源项目的源码实现，具体操作如下：

- \$\{cann\_version\}：表示CANN包版本号。
- \$\{arch\}：表示CPU架构，如aarch64、x86_64。

## 获取并安装软件包

如需修改某项目算子的源码，请确保已完成NPU驱动和固件、CANN开发套件包和`CANN-ops-base-${cann_version}−linux.${arch}.run`包安装。

1. 前提条件。

    参考《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》，按要求完成NPU驱动和固件、`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`软件包的获取和安装。

2. 安装`CANN-ops-base-${cann_version}−linux.${arch}.run`包。

    ```bash
    # 1.ops-base项目源码下载，以master分支为例
    git clone https://gitcode.com/cann/ops-base-dev.git
    # 2.进入项目根目录编译生成run包，默认在根目录build_out目录下
    bash build.sh
    # 3.安装编译包，${install_path}需与toolkit包指定路径一致
    ./CANN-ops-base-cannversion−linux.{cann_version}-linux.cann 
    version−linux.{arch}.run --full --install-path=${install_path}/ascend-toolkit
    ```

## 安装依赖

   开源项目的源码编译用到的依赖如下，请确保已安装并且满足版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.8）
   - dos2unix
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   - 项目使用的python依赖包，具体参见项目根目录中requirements.txt，安装命令如下：
     ```bash
     pip3 install -r requirements.txt
     ```
## 下载源码
通过`git`命令下载待修改项目的源码：
```bash
# 源码下载，以master分支为例
git clone https://gitcode.com/cann/${ops_project}.git
```
\$\{ops\_project\}表示待修改项目（如ops-math、ops-nn等）。

## 配置环境变量

根据实际场景，选择合适的命令。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 指定路径安装
# source ${install-path}/ascend-toolkit/set_env.sh
```

## 编译执行

基于商发版CANN开发套件包修改算子源码时，需使用**自定义算子包**方式编译和安装。

1. **编译自定义算子包。**

    进入待修改项目的根目录，执行如下编译命令：
    
    ```bash
    # 方式1：编译所有算子
    bash build.sh
    # 方式2：编译指定算子，如op1、op2
    bash build.sh -n "op1;op2"
    ```
    - -n（可选）：**仅编译部分算子设置**。"op1;op2"表示待编译的算子，多个算子之间使用英文分号";"分隔并使用引号。
    
    若提示如下信息，则说明编译成功。
    ```bash
    Self-extractable archive "CANN-ops-transformer-${cann_version}-linux.${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包。**
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh # 设置CANN开发套件包环境变量，以root用户默认路径为例，如已设置，则请忽略该操作
    ./CANN-custom_ops-<cann_version>-linux-${arch}.run
    ```

    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}需提前在环境变量配置，默认路径为`usr/local/Ascend`。
