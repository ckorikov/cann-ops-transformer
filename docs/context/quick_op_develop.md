# 算子开发

开发算子前，请先了解项目中已有算子，查看算子`README`了解算子的功能、参数规格和约束等。

若算子不满足业务场景诉求，您可以参考[开发指南](#开发指南)，实现符合业务场景的新算子。

## 算子列表

[算子列表](./op_list.md)

## 目录结构

开发算子前，建议您先熟悉项目目录结构，关键目录如下：

```
├── cmake                          # 项目工程编译目录
├── common                         # 项目公共头文件和公共源码
├── attention                      # attention类算子
├── docs                           # 项目文档介绍
├── example                        # 使用通用算子开发和调用示例
├── ...
├── moe                            # moe类算子
├── posembedding                   # posembedding类算子
│   ├── interleave_rope            # interleave_rope算子所有交付件，如Tiling、Kernel等
│   │   ├── CMakeLists.txt         # 算子编译配置文件
│   │   ├── docs                   # 算子说明文档
│   │   ├── examples               # 算子使用示例
│   │   ├── op_graph               # 算子构图相关目录
│   │   ├── op_host                # 算子信息库、Tiling、InferShape相关实现目录
│   │   │   └── op_api             # 算子aclnn接口实现目录
│   │   ├── op_kernel              # 算子kernel目录
│   │   └── README.md              # 算子说明文档
│   ├── ...
│   └── CMakeLists.txt             # 算子编译配置文件
├── scripts                        # 脚本目录，包含自定义算子、kernel构建相关配置文件
├── tests                          # 测试用例目录
├── CMakeLists.txt
├── README.md
├── build.sh                       # 项目工程编译脚本
├── install_deps.sh                # 安装依赖包脚本
└── requirements.txt               # 本项目需要的第三方依赖包
```

## 开发指南

> 关于算子运行在AI Core的详细介绍请参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。

- 若自定义算子运行在AI Core上，其开发过程和涉及的交付件如下：

    [AI Core算子开发指南](./aicore_develop_guide.md)