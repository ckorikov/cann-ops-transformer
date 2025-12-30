# 简介

CANN Simulator是一款面向算子开发场景的SoC级芯片仿真工具，用于分析运行在AI仿真器上的AI任务在各阶段的精度和性能数据（如指令执行情况等）。该工具有助于用户进行深度性能调优，使研发人员在无法获取或芯片资源紧缺的情况下，也能获得与真实芯片几乎一致的验证效果和性能反馈。

# 主要功能

该工具与板上运行保持二进制兼容（同一 kernel可同时在仿真和AI处理器执行），主要用途如下：
* 精度仿真：输出bit级精度结果，协助用户完成算子的精度验证。
* 性能仿真：输出指令流水图，协助用户定位算子性能瓶颈问题。

# 使用前准备

## 使用约束

* 工具推荐环境配置：CPU 16核，内存32GB以上。
* 本文中举例路径均需要确保运行用户具有读或读写权限。
* 出于安全性和权限最小化考虑，建议使用普通用户权限执行本工具，避免使用root等高权限账户。
* 本工具依赖CANN软件包，在使用前请先安装CANN软件包，无需安装赖驱动和固件，并通过source命令执行CANN的set_env.sh环境变量文件。为确保安全，执行source命令后请勿修改set_env.sh中涉及的环境变量。
* 用户应遵循最小权限原则，例如，给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。
* 本工具为开发工具，不建议在生产环境使用。
* 工具的仿真功能仅支持单卡场景，无法仿真多卡环境，代码中只能设置为0卡。若修改可见卡号，将导致仿真失败。
* 仿真环境仅支持AI Core计算类算子（不支持MC2和HCCL类型的算子）。
* 目前不支持arm环境仿真。

## 环境准备

CANN Simulator集成在CANN toolkit包里，参考[环境部署](../context/quick_install.md)中的软件包安装 -> 安装社区版CANN toolkit包章节

# 快速开始

## 算子编译

下文将以Add矢量算子为例对Kernel直调算子的仿真进行详细说明，开发者进行算子开发的步骤如下：

* 完成算子kernel侧实现。
* 编写算子调用应用程序main.cpp。
* 编写CMake编译配置文件CMakeLists.txt。

```
├── add_custom.cpp               --- kernel侧算子实现
├── cmake
│   └── npu_lib.cmake             --- kernel侧cmake编译
├── CMakeLists.txt                  --- 算子调用程序cmake编译
└── main.cpp                          --- 算子调用程序
```

### 算子Kernel侧实现

创建add_custom.cpp文件，可参考如下实现完成Ascend C算子实现文件的编写。

```
/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}

void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
{
    add_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}

```

### 算子调用应用程序

创建main.cpp文件，下面代码以固定shape的add_custom算子为例，x=i，y=i*2，打印前10的z的值。您在实现自己的应用程序时，需要关注由于算子核函数不同带来的修改，包括算子核函数名，入参出参的不同等，合理安排相应的内存分配、内存拷贝和文件读写等，相关API的调用方式直接复用即可。

```
/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "acl/acl.h"

#include <cstdio>
#include <fstream>
#include <iostream>

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

extern void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

	// 构造固定输入数据（FP16 类型）
    aclFloat16* xData = reinterpret_cast<aclFloat16*>(xHost);
    aclFloat16* yData = reinterpret_cast<aclFloat16*>(yHost);

    for (size_t i = 0; i < 8 * 2048; ++i) {
        xData[i] = aclFloatToFloat16(static_cast<float>(i));         // x[i] = i
        yData[i] = aclFloatToFloat16(static_cast<float>(i * 2));     // y[i] = 2*i
    }

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    add_custom_do(blockDim, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
	aclFloat16* zData = reinterpret_cast<aclFloat16*>(zHost);
    std::cout << "First 10 output values:\n";
    for (int i = 0; i < 10; ++i) {
        float val = aclFloat16ToFloat(zData[i]);
        std::cout << "z[" << i << "] = " << val << std::endl;
    }

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}

```

### CMake编译配置文件编写

简化的编译流程图如下图所示：将算子核函数源文件编译生成kernel侧的库文件（*.so或*.a库文件）；编译main.cpp（算子调用应用程序）时依赖上述头文件，将编译应用程序生成的目标文件和kernel侧的库文件进行链接，生成最终的可执行文件

创建CMakeLists.txt文件，如下是CMake示例，通常情况下不需要开发者修改

```
cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(SOC_VERSION "Ascend910_9599" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH}
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

# ${KERNEL_FILES} are used to compile library, push files written by ascendc in ${KERNEL_FILES}.
# ref to cmake/npu.cmake ascendc_library, cmake/cpu.cmake add_library
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/add_custom.cpp)

include(cmake/npu_lib.cmake)
add_executable(ascendc_kernels_bbit ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_compile_options(ascendc_kernels_bbit PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    host_intf_pub
    ascendc_kernels_npu
)

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

```

创建cmake文件夹，以及包含的npu_lib.cmake文件，编译出kernel侧的库文件，参考如下

```
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

```

### 执行编译

在算子开发目录执行编译命令

```
mkdir build && cd build && cmake .. && make -j
```

编译完成后，在build目录会生成对应的可执行文件ascendc_kernels_bbit和lib/libascendc_kernels_npu.so，将对应的so库路径加入到LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH  -- pwd为当前算子编译后的的build目录
```

## 执行仿真命令

```
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report
```

仿真工具执行日志文件在add_example/build/cannsim_*目录，执行日志文件为

```
cannsim.log
```

从仿真工具日志文件可以看到示例中的前10结果打印信息：

```
First 10 output values:
z[0] = 0.000000
z[1] = 3.000000
z[2] = 6.000000
z[3] = 9.000000
z[4] = 12.000000
z[5] = 15.000000
z[6] = 18.000000
z[7] = 21.000000
z[8] = 24.000000
z[9] = 27.000000
```

## 查看性能流水

仿真性能流水文件在本项目`examples/add_example/build/cannsim_*/report目录，流水相关文件为：

```
trace_core0.json
```

在Chrome浏览器中输入“chrome://tracing”地址，并将生成的指令流水图文件（trace_core0.json）拖到空白处打开，具体参数介绍参考“仿真结果解析”章节。

# 仿真执行说明

## 命令功能

在仿真环境中执行应用程序。

## 命令格式

cannsim record [options] user_app --user_options

## 参数说明

表1 仿真执行参数说明

|参数|可选/必选|说明|
| --- | --- | --- |
|-s <value> 或 --soc_version <value> [options]参数 | 必选 | 指定模拟目标芯片版本（如：Ascend950）。|
|-o <value> 或 --output <value> [options]参数 | 可选| 生成文件所在路径，可配置为绝对路径或者相对路径，并且执行工具的用户需要具有读写权限。如果未指定路径，则默认在当前目录下保存数据。|
|-g 或 --gen-report[options]参数 | 可选 | 启用仿真完成后是否进行自动解析，并生成分析报告。默认不自动解析。|
|user_app|必选|算子可执行文件。|
--user_options|可选|算子可执行文件的运行参数。|

## 使用示例

1. 完成算子开发和编译。
2. 执行仿真命令，可参考以下使用示例

    ```
    方式一： 启用仿真，并将输出保存至 ./output 目录，/path/to/app 为算子程序
    $ cannsim record /path/to/app -o ./output -s Ascend950

    方式二：启用仿真并生成报告，用于后续性能分析
    $ cannsim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```

3. 命令完成后，会在默认路径或指定的“output”目录下生成以“cannsim_{timestamp}_${user_app}”命名的文件夹，结构示例如下：

    ```
    ├─cannsim_{timestamp}_${user_app}
    ├── cannsim.log
    ├── log
    │   ├── AIC_0_0_0_0_ChiWrap.log0
    │   ├── ccum_0_0_2.txt0
    │   ├── hha_0_0_0_states.log
    │   ├── L2Buf_0_0_0_0.txt0
    │   ├── L2cache_stats.log
    │   ├── lpddr_state0.log
    │   ├── ManyRing_0_0_0_0DatPerf.txt0
    │   ├── sdmam_0_0_0_debug.log0
    │   ├── sllc_chip_0_die_0_1_0_0_perf.log
    │   ├── STARS_ChiWrapper_0_1.log0
    │   ├── Tg_Log_File_0_0.txt
    │   ├── UB_0_0_2_0_ChiWrap.log0
    │   └── ub_log
    │       ├── UB_0_0_2_BA_0_statis.log
    ├── log_ca
    │   ├── core0.cubecore0_su_perf_summary_log
    │   ├── core0_summary_log
    │   ├── core0.wrapper_log.dump
    │   ├── mcu_log.dump
    │   ├── stars_log0_0.dump
    │   └── stars_log0_1.dump
    ```

4. 用户可以获取算子执行结果，并进行精度的对比，结果展示在cannsim.log，示例如下

    以下输出仅为AscendC单算子直调精度比较结果举例，因版本不同略有差异，请以实际输出为准。

    ```
    INFO:root:[INFO] compare data case[ case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```

5. 查看算子指令流水图，参考仿真结果解析。

# 仿真结果解析说明

## 命令功能

生成可视化的指令流水图。

## 命令格式

cannsim report [options]

## 参数说明

表1 仿真结果解析参数说明

|参数 | 可选/必选 | 说明|
| --- | --- | --- |
|-e <value> 或 --export <value> [options]参数 | 必选 | 原始结果文件目录，需指定为仿真执行后生成的结果目录，指定到cannsim_{timestamp}_${user_app}层，可配置为绝对路径或者相对路径，并且工具执行用户具有可读写权限。|
|-o  或 --output  [options]参数 | 可选 | 解析结果输出目录，可配置为绝对路径或者相对路径，且执行用户需具有读写权限。若未指定路径，默认在当前目录下保存数据。如果生成的结果文件与现有文件同名，则会覆盖原有文件。|
|-n 或 --core-id  [options]参数 | 可选 | 指定生成指令流水的核ID，不指定默认生成0核的指令流水。配置的格式如下：生成所有核的流水，配置为‘all’。指定核ID的范围，如：‘0-1’。指定单核ID，如‘5’。|

## 使用示例

1. 参考仿真执行执行算子仿真，对比输出示例，确保对应的结果执行正确。
2. 执行仿真结果解析命令，可参考以下执行用例。

    ```
    在当前目录下生成性能分析报告（默认仅分析核0）
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} 

    在指定目录下生成核0、核1、核11、核12的性能分析报告
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} -o /path/to/report -n ‘0-1, 11-12’
    ```

3. 命令执行完后，会在output配置的目录下生成对应的流水文件，文件格式为json格式，输出结果示例如下：

    ```
    trace_core0.json
    trace_core1.json
    ...
    ```

4. 仿真结果查看
    在Chrome浏览器中输入“chrome://tracing”地址，并将通过生成指令流水图文件（trace.json）拖到空白处打开，键盘上输入快捷键（W：放大，S：缩小，A：左移，D：右移）可进行查看。
    ![指令流水图](../figures/指令流水图%20.png)

    表2 关键字段说明

    |字段名|字段含义|
    | --- | --- |
    |VECTOR|向量运算单元。|
    |SCALAR|标量运算单元。|
    |Cube|矩阵乘运算单元。|
    |MTE1|数据搬运流水，数据搬运方向为：L1 ->{L0A/L0B, UBUF}。|
    |MTE2|数据搬运流水，数据搬运方向为：{DDR/GM, L2} ->{L1, L0A/B, UBUF}。|
    |MTE3|数据搬运流水，数据搬运方向为：UBUF -> {DDR/GM, L2, L1}、L1->{DDR/L2}。|
    |FIXP|数据搬运流水，数据搬运方向为：FIXPIPE L0C -> OUT/L1。（仅 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 支持展示）|
    |FLOWCTRL|控制流指令。|
    |ICACHELOAD|查看未命中的ICache。|

# 查询帮助信息

## 命令功能

查询工具帮助信息。

## 命令格式

查询工具帮助信息：

```
cannsim --help
```

查询工具record 子命令的帮助信息：

```
cannsim record --help
```

查询工具report子命令的帮助信息：

```
cannsim report --help
```

## 参数说明

无

## 使用示例

1. 登录Host侧服务器。
2. 执行以下命令。

    ```
    cannsim --help
    ```

## 输出说明

```
Usage: cannsim [OPTIONS] COMMAND [ARGS]...

Command-line tool for performance simulation analysis on Ascend hardware.
The simulation emulates real Ascend hardware behavior—including compute
units, memory hierarchy, and scheduling—enabling accurate performance
modeling without physical devices.

Examples:
$ cannsim record ./app -s Ascend910B -o ./output
$ cannsim report -e ./output/sim -o ./output/trace.json

Note:
- Input app must be a valid AscendOps-built executable.
- Output directories are auto-created.
- `trace.json` is compatible with Chrome tracing tools.
- Advanced reporting features (e.g., HTML, diagrams) are planned but not yet available.

Options:
--help  Show this message and exit.

Commands:
record  Run user application in AscendOps simulation environment.
report  Command-line interface for generating performance reports.

```
