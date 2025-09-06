# 算子开发样例
本章主要以`AddExample`算子为例，介绍算子开发中的目录结构、各交付件及其命名、写法，帮助开发者快速上手算子开发。

## 目录结构介绍

```
├── graph_plugin                                    // 图融合相关
│   └── add_example_proto.h                         // 算子原型
├── op_host                                         // 算子Host侧实现目录
│   ├── config                                      // 配置相关目录
│   │   └─ ascend910b                               // 910b 相关配置
│   │       ├─ add_example_binary.json              // 910b 二进制配置
│   │       └─ add_example_simplified_key.ini       // 910b kernel编译 simplified_key_mode 值配置
│   ├── add_example_def.cpp                         // 算子信息库
│   ├── add_example_infershape.cpp                  // 算子InferShape实现
│   ├── add_example_tiling.cpp                      // 算子tiling实现
│   └── add_example_tiling.h                        // 算子tiling头文件
└── op_kernel                                       // 算子Device侧Kernel实现目录
    ├── add_example.cpp                             // 算子kernel入口文件
    └── add_example.h                               // 算子kernel实现文件
```

## 算子交付件介绍

算子区分图模式，单算子模式。

`单算子`指的是在深度学习模型中，一个单独的计算操作（Operator），如`Add`,`Relu`，每个算子都有明确的输入核输出。

`图模式`是指将整个模型或者部分模型表示为一个计算图，然后一次性编译并执行整个图。图模式会将算子连接构成图。

下表标注算子开发中涉及的各交付件及其在两种模式是否必须。`1`表示必须，`0`表示非必须。
<table>
<tr>
<th>交付件</th>
<th>图模式</th>
<th>单算子模式</th>
</tr>
<tr>
<td>算子原型</td>
<td>1</td>
<td>0</td>
</tr>
<tr>
<td>算子信息库</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>tiling</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>infershape</td>
<td>1</td>
<td>0</td>
</tr>
<tr>
<td>kernel</td>
<td>1</td>
<td>1</td>
</tr>
<tr>
<td>二进制配置</td>
<td>0</td>
<td>1</td>
</tr>
</table>

注册算子类型后，框架会根据算子类型获取算子注册信息，同时在编译和运行时按照一定的规则匹配算子实现文件名称和kernel侧核函数名称。为了保证正确匹配，算子类型、算子实现文件名称和核函数名称需要遵循一定的规则，算子类型由大驼峰命名，其他相关文件以小写字母加下划线的形式对应，不同交付件补充以特定后缀或者前缀，如`_def`,`_proto`,`_tiling` 等。具体在每个章节介绍。

### 算子原型

算子的原型定义了包括算子的输入、输出、属性以及对应的数据类型。会向GE注册该算子的原型，告知GE对应类型的算子应该具备哪些输入、输出与属性；同时相当于定义了一个op::xxx的Class，开发者可以include该原型头文件，然后实例化该Class进行IR模型构建，如下所示：
```
conv = op::AddExample()
```

算子名称以大驼峰命名，如`AddExample`。

原型文件以`{算子名小写下划线格式}_proto.h`命名，如`add_example_proto.h`。

样例参考[add_example_proto.h](./op_graph/add_example_proto.h)。文档参考[算子原型定义](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/basicdataapi/atlasopapi_07_00527.html)。


### 算子信息库

算子信息库定义主要描述算子的输入输出、属性等信息以及算子在AI处理器上相关实现信息，并关联tiling实现等函数。算子信息库通过自定义的算子类来承载，该算子继承自`OpDef`类。完成算子信息库定义等操作后，需要调用`OP_ADD`接口，传入算子类型（自定义算子类的类名），进行算子信息库注册。

算子名称以驼峰法命名，如`AddExample`。

原型文件以`{算子名小写下划线格式}_def.cpp`命名，如`add_example_def.cpp`。

样例参考[add_example_def.cpp](./op_host/add_example_def.cpp)。文档参考[算子信息库定义](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0062.html)。

### InferShape

InferShape函数的原型是确定的，其接受一个InferShapeContext类型作为输入，在此context上，可以获取到输入、输出的shape指针等内容（算子原型定义上的输入、输出、属性信息）。InferShape成功后，返回ge::GRAPH_SUCCESS，其他返回值被认为推导失败。推导失败后，执行过程结束退出。

InferShape文件以`{算子名小写下划线格式}_infershape.cpp`命名，如`add_example_infershape.cpp`。


样例参考[add_example_infershape.cpp](./op_host/add_example_infershape.cpp)。文档参考[InferShape](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/basicdataapi/atlasopapi_07_00115.html)。


### Host侧tiling实现

由于NPU中AI Core内部存储无法完全容纳算子输入输出的所有数据，需要每次搬运一部分输入数据进行计算然后搬出，再搬运下一部分输入数据进行计算，这个过程就称之为`Tiling`。切分数据的算法称为Tiling算法或者Tiling策略。根据算子的shape等信息来确定数据切分算法相关参数（比如每次搬运的块大小，以及总共循环多少次）的计算程序，称之为Tiling实现，也叫Tiling函数（Tiling Function）。由于Tiling实现中完成的均为标量计算，AI Core并不擅长，所以我们将其独立出来放在Host侧CPU上执行。

tiling文件以`{算子名小写下划线格式}_tiling`加文件后缀格式命名，如`add_example_tiling.h`，`add_example_tiling.cpp`，如果存在多个tiling文件，在中间插入特定场景标识，如区分输入format类型时，将tiling文件分别命名`add_example_nchw_tiling`或者`add_example_nhwc_tiling`加文件名后缀`.h`或者`.cpp`。

样例参考[add_example_tiling.cpp](./op_host/add_example_tiling.cpp)。文档参考[Host侧tiling实现](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0064.html)。

### Device侧Kernel实现

Kernel实现即算子核函数实现，在Kernel函数内部通过解析Host侧传入的Tiling结构体获取Tiling信息，根据Tiling信息控制数据搬入搬出Local Memory的流程；通过调用计算、数据搬运、内存管理、任务同步API，实现算子逻辑。其核心逻辑基本上都为计算密集型任务，需要在NPU上执行。

kernel入口文件以`{算子名小写下划线格式}.cpp`命名，如`add_example.cpp`。kernel核心实现在`.h`头文件中编写。

样例参考[add_example.h](./op_kernel/add_example.h)。文档参考[Kernel侧算子实现](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0063.html)。

### 二进制相关配置

算子的二进制包由编译期直接生成，模型在运行时直接使用对应二进制文件，从而减少模型运行时编译时间。在二进制json配置中，会指定算子的输入、输出以及属性等信息。框架的打包工具在打包过程中，会根据该配置生成对应的算子`.json`,`.o`文件，模型运行时，会直接加载并匹配相应的`.o`文件进行执行。

二进制配置文件以`{算子名小写下划线格式}_binary.json`命名放在支持的`soc版本`目录下，如`op_host/config/ascend910b/add_example_binary.json`

样例参考[add_example_binary.json](./op_host/config/ascend910b/add_example_binary.json)

### 单算子API接口开发

单算子API调用方式，是指直接调用单算子API接口，基于C语言的API执行算子。完成自定义算子编译后，会自动生成单算子API，可以直接在应用程序中调用。

文档参考[单算子API调用](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0070.html)

### 编译运行

#### 必要配置项
当前依赖如下配置，完成算子编译，运行

1、`ops-math-dev/kernel/binary_config/ascendc_config.json` 中注明算子`soc版本`及实现模式，如：
```json
    {"name":"AddExample", "compute_units": ["ascend910b"], "auto_sync":true, "impl_mode" : "high_performance"},
```

2、`ops-math-dev/kernel/binary_config/binary_config.csv`中配置算子二进制json文件名，如：
```json
AddExample,add_example.json,6
```
#### 编译装包

1、将样例算子由 `ops-math-dev/example/add_example`移到 `ops-math-dev/math/add_example`

2、安装`ops-base`包，参考[ops-base-dev](https://gitcode.com/cann/ops-base-dev)

3、编译自定义算子包

```bash
code/ops-math-dev#  bash build.sh --ops=add_example --package --soc=ascend910b
```

4、安装自定义算子包
```bash
code/ops-math-dev/build_out#  ./custom_ops_math_ubuntu_x86_64.run
```

#### 运行样例验证

单算子调用样例参考 `ops-math-dev/example/add_example/examples/eager_mode/test_aclnn_add_example.cpp`

执行该目录下的`run.sh` ，最终打印如下即可证明该算子执行成功。

```
mean result[2045] is: 2.00000
mean result[2046] is: 2.00000
mean result[2047] is: 2.00000
```

