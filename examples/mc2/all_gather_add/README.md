# ALLGatherAdd
> 说明：本项目介绍一个简单的通算融合算子AllGatherAdd，并阐述如何配合社区版CANN开发套件包对该算子进行调用。
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：完成AllGather通信与Add计算融合。
- 计算公式：

    $$
    gatherOut=AllGather(a0, a1)
    $$
    $$
    c[i]=gatherOut[i] + b[i]
    $$

- 算子语义示例：(两张卡参与计算，分别为rank0和rank1)

  - 输入:  
    rank0_a = [1,2,3];  
    rank1_a = [4,5,6,7,8,9];  
    rank0_b = [10,11,12];  
    rank1_b = [13,14,15,16,17,18];  

  - 输出：  
    rank0_c = AllGatherAdd(rank0_a, rank1_a, rank0_b)  
              = AllGather(rank0_a,rank1_a) + rank0_b  
              = [1,2,3,10,11,12] + [4,5,6,7,8,9]  
              = [5,7,9,17,19,21]  

    rank1_c = AllGatherAdd(rank0_a, rank1_a, rank1_b)  
              = AllGather(rank0_a,rank1_a) + rank1_b  
              = [1,2,3,10,11,12] + [13,14,15,16,17,18]  
              = [14,16,18,26,28,30]  

## 参数说明


<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>输入</td>
      <td>公式中的输入a。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>公式中的输入b。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输出</td>
      <td>公式中的输出c。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gather_out</td>
      <td>输出</td>
      <td>公式中的输出gatherOut。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td><li>Host侧标识通信域的字符串，通信域名称。</li><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>可选属性</td>
      <td><li>通信数据切分数，即总数据量/单次通信量。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rank_size</td>
      <td>可选属性</td>
      <td><li>通信域里面的卡数。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
* 当前该示例算子仅支持固定shape: a(240, 256)，b(240 * 2, 256)。 
* 所有输入不支持空tensor场景。
* commTurn当前版本仅支持输入1。
## 调用说明

调用本算子前，请确保已本地下载代码仓，并安装好如下基础依赖、NPU驱动和固件已安装。

本项目源码编译用到的依赖如下，请参考[算子调用](../../../docs/invocation/quick_op_invocation.md)文档中的**前提条件**和**环境准备**章节，完成环境依赖的下载和CANN包的准备，其中，**环境准备**章节中的**安装社区版CANN ops-math包**部分可以跳过。

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
    - --ops：填写本示例算子名称 all_gather_add。
     
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

        如需执行算子样例，需将自定义算子包安装在默认路径下。执行算子样例后[test_aclnn_all_gather_add.cpp](examples/test_aclnn_all_gather_add.cpp)文件会按照固定shape随机生成测试数据调用算子，并打印算子与golden的对比执行结果，示例结果如下：
    
        ```
        device_0 aclnnAllGatherAdd execute successfully.
        device_1 aclnnAllGatherAdd execute successfully.
        device_0 aclnnAllGatherAdd golden compare successfully.
        device_1 aclnnAllGatherAdd golden compare successfully.
        ```