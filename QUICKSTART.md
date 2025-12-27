# 算子开发快速入门：基于ops-transformer仓

本指南旨在帮助你快速上手基于CANN和`ops-transformer`算子仓库的算子开发。通过一个从零开始的完整流程，你将学会如何将一个算子代码编译部署、运行验证，并进一步完成修改、调试和性能分析。

## 目录导读
1.  **[环境安装](#一环境安装)**：通过Docker快速搭建算子开发环境。
2.  **[编译部署](#二编译部署)**：编译自定义算子包并进行安装，快速调用算子example验证。
3.  **[算子开发](#三算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。
4.  **[算子调试与调优](#四算子调试与调优)**：掌握定位逻辑错误的调试方法和分析性能瓶颈的调优工具。
5.  **[算子验证](#五算子验证)**：学习如何修改测试用例，以验证算子在不同输入下的功能正确性。

完成以上步骤，你将对自定义算子开发的全流程有一个清晰的实践认知。

## 一、环境安装

算子开发的第一步是准备一个包含CANN及算子仓前置依赖的环境，使用Docker镜像是最高效的方式。

**前提条件**：
*   **Docker环境**：宿主机已安装Docker引擎（版本1.11.2及以上）。
*   **驱动与固件**：宿主机已安装昇腾NPU的驱动与固件。安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

### 1. 下载镜像
从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)拉取已预集成CANN软件包及`ops-transformer`所需依赖的镜像。

*   **操作步骤**：
    1.  以root用户登录宿主机。
    2.  执行拉取命令（请根据你的宿主机架构选择）：
        ```bash
        # 示例：拉取ARM架构的CANN开发镜像
        docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        # 示例：拉取X86架构的CANN开发镜像
        docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        > **注意**：镜像文件较大，下载需要一定时间。

### 2. Docker运行
拉取镜像后，需要以特定参数启动容器，以便容器内能访问宿主的昇腾设备。

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```
| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--device /dev/davinci0` | 核心：将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行 `npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。|
| `--device /dev/davinci_manager` | 映射NPU设备管理接口。 |  |
| `--device /dev/devmm_svm` | 映射设备内存管理接口。 |  |
| `--device /dev/hisi_hdc` | 映射主机与设备间的通信接口。 |  |
| `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载设备容器管理接口（DCMI）相关工具和库。 | |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键挂载：将宿主机的NPU驱动库映射到容器内。 | |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | 挂载驱动版本信息文件。 | |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载CANN软件安装信息文件。 | |
| `-it` | `-i`（交互式）和 `-t`（分配伪终端）的组合参数。 | |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/8.5.0-910b-ubuntu22.04-py3.10-ops-x86` | 指定要运行的Docker镜像。 |请确保此镜像名和标签（tag）与你通过`docker pull`拉取的镜像完全一致。 |
| `bash` | 容器启动后立即执行的命令。 | |

### 3. 检查环境
进入容器后，验证环境和驱动是否正常。

*   **检查NPU设备**：
    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```
*   **检查CANN安装**：
    ```bash
    # 查看CANN Toolkit版本信息
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

**环境就绪意味着**：你已经拥有了一个“开箱即用”的算子开发沙箱。接下来，需要在这个沙箱里验证从源码到可运行算子的完整工具链。

## 二、编译部署

本阶段的目的是**快速走通标准流程**，验证你的开发环境能否成功地将算子源代码编译、打包、安装并运行。我们以仓库内置的`AddExample`算子作为实践对象。

### 1. 拉取ops-transformer仓库代码

在容器内获取算子源代码。
```bash
git clone https://gitcode.com/cann/ops-transformer.git
cd ops-transformer
```

### 2. 指定add_example算子编译自定义算子包

以AddExample算子为例，进入项目根目录，编译指定的AddExample算子。
```bash
# 编译命令格式：bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>
bash build.sh --pkg --soc=ascend910b --ops=add_example
```
> 成功标志：在build_out目录下生成名为 cann-ops-transformer-*_linux-*.run 的自定义算子安装包。

### 3. 安装自定义算子包
```bash
./build_out/cann-ops-transformer-*_linux-*.run
```
自定义算子包安装在`${ASCEND_HOME_PATH}/opp/vendors`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录，可提前在环境变量中配置。

### 4. 配置环境变量

将自定义算子包的路径加入环境变量，确保运行时能够找到。
```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

### 5.快速验证：运行算子样例

执行内置的算子样例，验证算子功能是否正常。

```bash
# 运行命令格式：bash build.sh --run_example <算子名> <运行模式> <包模式>
bash build.sh --run_example add_example eager cust --vendor_name=custom
```
> 预期输出：终端将打印出加法计算结果，表明算子已成功部署并正确执行。
```
add_example result[0] is: 2.000000
add_example result[1] is: 2.000000
add_example result[2] is: 2.000000
add_example result[3] is: 2.000000
add_example result[4] is: 2.000000
add_example result[5] is: 2.000000
add_example result[6] is: 2.000000
add_example result[7] is: 2.000000
```

成功运行AddExample算子，意味着你已经掌握了标准流程。接下来，我们尝试修改这个算子的核函数代码，这是理解算子开发精髓、学习如何添加自定义逻辑的关键一步。

## 三、算子开发
现在，你将从一个“使用者”变为“修改者”。通过给AddExample算子的Kernel函数添加一行简单的打印信息，来体验算子代码开发、编译验证的完整微型闭环。

### 1. 了解算子工程结构
在修改前，先快速了解add_example算子的关键文件：

内置算子的工程目录结构如下，核心代码为`op_host`下`${op_name}_tiling.cpp`文件，以及`op_kernel`下`${op_name}.h`文件：

``` text
add_example/
├── op_kernel/               # Device侧核函数实现（运行在NPU上）
│   ├── add_example.cpp      # Kernel入口
│   └── add_example.h        # **核心计算逻辑实现文件（本次修改目标）**
└── op_host/                 # Host侧实现（运行在CPU上）
    └── add_example_tiling.cpp # 数据切分策略
```

### 2. 在核心kernel实现文件中增加打印：
找到AddExample算子的核心kernel实现文件[add_example.h](./examples/add_example/op_kernel/add_example.h)，尝试在Init函数中增加打印：

```
__aicore__ inline void AddExample<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData)
{
    // ==== 在此处添加一行调试打印 ====
    AscendC::PRINTF("This is AddExample Kernel Init.\n");
    
    blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    // ... 后续原有代码 ...
}
```
保存修改。

### 3.重新编译、部署与验证：

重复[编译部署](#二编译部署)章节中的第2至第5步：
1. **重新编译**：`bash build.sh --pkg --soc=ascend910b --ops=add_example`。
2. **重新安装**：运行新生成的.run安装包。
3. **重新验证**：`bash build.sh --run_example add_example eager cust --vendor_name=custom`。
4. **成功标志**：在运行结果中，除了计算结果，你应能看到打印的 “This is AddExample Kernel Init.” 信息。

完成这一步，你就实现了第一次算子代码变更！ 这证明了你可以修改NPU上执行的核函数代码。当开发更复杂的算子时，你必然会遇到需要定位的逻辑问题或性能瓶颈，因此接下来需要掌握调试和调优的方法。

## 四、算子调试与调优
### 1. 调试：定位逻辑与数据问题
算子运行过程中，如果出现算子执行失败、精度异常等问题，可以打印各阶段信息，如Kernel中间结果，进行问题分析和定位。

以`AddExample`算子为例，常见调试方法如下：

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。
  
  ```c++
  blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) ?
        ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) : 1;
  // 打印当前核计算Block长度
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```
* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // 打印zLocal Tensor信息
  DumpTensor(zLocal, 0, 128);
  ```
### 2. 调优：使用msprof分析性能

当算子功能正确后，需要关注其在NPU上的执行效率，可通过[msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0008.html)性能分析工具分析算子各运行阶段指标数据（如吞吐率、内存占用、耗时等），从而确定问题根源，并针对性地优化。

本章以`AddExample`自定义算子为例，通过采集算子上板运行时各项流水指标分析算子Bound场景。

* **生成可执行文件**：
    ```
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

    调用`AddExample`的example样例，生成算子可执行文件（test_aclnn_add_example），所在目录为本项目`ops-transformer/build/`。

* **采集性能数据**：

进入`AddExample`算子可执行文件目录`ops-transformer/build/`，执行如下命令：

```bash
msprof --application="./test_aclnn_add_example"
```
采集结果在本项目`ops-transformer/build/`目录，msprof命令执行完成后，会自动解析并导出性能数据结果文件，详细内容请参见：[msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251)

掌握了调试和调优手段，你的算子开发能力将更加全面。最后，为了确保算子的通用性，需要学习如何构建不同的测试用例对其进行验证。

## 五、算子验证

一个健壮的算子需要应对各种合法的输入。通过修改example测试用例中的输入数据，可以验证算子在多种场景下的功能正确性。

### 1. 修改测试输入
找到并编辑`AddExample`的[example](./examples/add_example/examples/test_aclnn_add_example.cpp)，修改输入张量的形状和数值。

- **修改输入、输出数据**：
修改输入、输出的shape信息，以及初始化数据，构造相应的输入、输出tensor。
```c++
int main() {
    // ... 初始化代码 ...
    
    // 修改前：shape = {32, 4, 4, 4}, 数值全为1
    // 修改后：将输入shape改为 {8, 8}，并填充不同的测试数据
    std::vector<int64_t> selfXShape = {8, 8};
    std::vector<float> selfXHostData(64); // 64 = 8*8
    // 可使用循环填充更有区分度的数据，例如递增序列
    for (int i = 0; i < 64; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // 填充0-9的循环值
    }
    // 同理修改selfY的输入...
    
    // ... 后续执行代码 ...
}
```
### 2. 重新编译并运行验证

1. 由于只修改了example测试代码，无需重新编译算子包。

2. 直接重新运行验证命令即可：`bash build.sh --run_example add_example eager cust --vendor_name=custom`

3. 观察输出结果是否符合运算的预期。

## 总结与进阶
至此，你已完整体验了基于`ops-transformer`仓库的算子开发入门流程：从搭建环境，到走通编译部署流程，再到亲手修改代码、调试、性能分析以及构造测试用例。
这是一个最简化的学习路径。若需深入每个环节，请参考以下详细指南：

- [环境部署](./docs/zh/context/quick_install.md)：更详细的环境搭建说明。
- [编译部署及算子调用](./docs/zh/invocation/quick_op_invocation.md)：深入了解编译参数与调用方式。
- [算子开发](./docs/zh/develop/aicore_develop_guide.md)：学习如何从零创建算子工程，实现Tiling和Kernel。
- [调试调优](./docs/zh/debug/op_debug_prof.md)：掌握更系统的调试技巧与性能优化方法。
- [贡献指南](CONTRIBUTING.md)：开发完成算子后，基于贡献指南完成代码上库。

