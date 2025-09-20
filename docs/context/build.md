# build参数说明

## 简介
build.sh是本项目的构建脚本，默认在项目根目录下，其作用是将源代码自动编译、链接和配置，最终生成可执行文件、库文件或其它可供安装或直接运行的目标文件。具体来说，脚本中通过配置不同参数实现多种功能，包含构建多种目标库（如：libophost_math.so）、编译算子包、执行单元测试等。


## 使用方法 
1. **配置环境变量**
   
   参考[快速入门 > 环境准备](./QuickStart.md#环境准备)完成环境变量配置。
   ```bash
   # 默认路径安装，以root用户为例
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
2. **构建命令格式**

   以编译算子包命令为例，样式如下，全量参数含义参见[参数说明](#参数说明)。
   ```bash
   bash build.sh --package --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op1,op2,...}]
   ```

## 参数说明
build.sh支持多种功能，可通过如下命令查看所有功能参数，请按实际需要选择。
```bash
bash build.sh --help
```

详细参数介绍待补充。