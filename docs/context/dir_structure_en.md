# Project Directory
The detailed directory hierarchy is as follows:

> Some of the directories listed in this chapter are optional; please refer to the actual deliverables for details. In particular, the **single-operator directory** varies depending on the scenario, and the specific details are as follows:
>
> - If the op_host directory is missing, it may be because the implementation of another operator's op_host is being called. For the calling logic, refer to the source code implementation in the op_api or op_graph directory of that operator. Alternatively, it could be that the kernel currently does not have an Ascend C implementation. If needed, developers are welcome to contribute this operator by referring to the [Contribution Guide](../../CONTRIBUTING.md).
> - If the op_kernel directory is missing, it may be because the implementation of another operator's op_kernel is being called. For the calling logic, refer to the source code implementation in the op_api or op_graph directory of that operator. Alternatively, it could be that the kernel currently does not have an Ascend C implementation. If needed, developers are welcome to contribute this operator by referring to the [Contribution Guide](../../CONTRIBUTING.md).
> - If the op_api directory is missing, it indicates that this operator is currently not supported for aclnn calls.
> - If the op_graph directory is missing, it indicates that this operator is currently not supported for graph mode calls.

```
├── cmake                                               # Project Engineering Compilation Directory
│   ├── aclnn_ops_transfomer.h.in                       # aclnn summary header file template
│   └── ...
├── common                                              # Project Public Header Files and Public Code
│   ├── CMakeLists.txt
│   ├── inc                                             # Public header file directory
│   └── src                                             # Public code directory
├── experimental                                        # User-defined operator storage directory
│   ├── attention                                       # Optional, directory for user-developed attention-type operators
│   │   └── CMakeLists.txt
│   ├── ffn                                             # Optional, directory for user-developed FNN class operators
│   │   └── CMakeLists.txt
│   ├── gmm                                             # Optional, directory for GMM operator developed by users.
│   │   └── CMakeLists.txt
│   ├── mc2                                             # Directory of mc2 operators developed by users. This parameter is optional.
│   │   └── CMakeLists.txt
│   ├── moe                                             # Optional, directory for MOE operator developed by users.
│   │   └── CMakeLists.txt
│   └── posembedding                                    # Optional, directory for user-developed posembedding class operators.
│       └── CMakeLists.txt
├── ${op_class}                                         # Operator classification, such as attention, ffn, and gmm operators.
│   ├${op_name}                                         # Operator project directory, where ${op_name} represents the operator name (in lowercase with underscores).
│   │   ├── CMakeLists.txt                              # Operator CMakeLists entry
│   │   ├── README.md                                   # Operator introduction documents
│   │   ├── docs                                        # Operator document directory
│   │   │   └── aclnn${OpName}.md                       # Operator aclnn API introduction documents. ${OpName} indicates the operator name (in upper camel case).
│   │   ├── examples                                    # Operator call example directory
│   │   │   ├── test_aclnn_${op_name}.cpp               # Example of calling the operator through aclnn
│   │   │   └── test_geir_${op_name}.cpp                # Example of calling the operator through geir
│   │   ├── op_graph                                    # Implementation of graph fusion
│   │   │   ├── CMakeLists.txt                          # CMakeLists file on the op_graph side
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType file, which implements operator data type inference.
│   │   │   ├── ${op_name}_proto.h                      # Operator prototype definition, which is used to identify operators during graph optimization and fusion.
│   │   │   └── fusion_pass                             # Operator fusion rule directory
│   │   ├── op_host                                     # Host-side implementation
│   │   │   ├── CMakeLists.txt                          # CMakeLists file on the host side
│   │   │   ├── config                                  # (Optional) Binary configuration file. If not configured, the project will be automatically generated.
│   │   │   │   ├── ${soc_version}                      # Binary information of the operator configured on the NPU. ${soc_version} indicates the NPU model.
│   │   │   │   │   ├── ${op_name}_binary.json          # Operator binary configuration file
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # Operator SimplifiedKey configuration information
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # Operator information library, which defines basic operator information, such as the name, input and output, and data type.
│   │   │   ├── ${op_name}_infershape.cpp               # Optional. InferShape implementation, which is used to derive the output shape based on the operator shape. If this file is not configured, the output shape is the same as the input shape.
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # Optional. Tiling optimization for some sub-scenarios. ${sub_case} indicates the sub-scenario. For example, ${op_name}_tiling_arch35 indicates the optimization for the arch35 architecture. If this file does not exist, it indicates that the operator does not have a specific tiling policy for the corresponding sub-scenario.
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # Optional. Header file used for tiling implementation in the ${sub_case} sub-scenario.
│   │   │   ├── ${op_name}_tiling.cpp                   # Optional. If this file does not exist, it indicates that there is no tiling implementation in the corresponding scenario (the tensor is divided into multiple blocks and parallel computing is performed by distinguishing data types).
│   │   │   ├── ${op_name}_tiling.h                     # Optional. Header file used for tiling implementation.
│   │   │   └── op_api                                  # Optional. Directory of the aclnn operator implementation file. If this file is not configured, the project will automatically generate it.
│   │   │       ├── aclnn_${op_name}.cpp                # Implementation file of the aclnn operator interface.
│   │   │       ├── aclnn_${op_name}.h                  # Header file of the aclnn operator interface.
│   │   │       ├── ${op_name}.cpp                      # Implementation file of the L0 operator interface.
│   │   │       ├── ${op_name}.h                        # Header file of the L0 operator interface.
│   │   │       └── CMakeLists.txt
│   │   │── op_kernel                                   # AI Core Operator Kernel Implementation on the Device Side
│   │   │   ├── ${sub_case}                             # (Optional) Directory used in the ${sub_case} sub-scenario
│   │   │   │   ├── ${op_name}_${model}.h               # Operator kernel implementation file. ${model} indicates the user-defined file name extension, which is usually the tiling template.
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # Optional. TilingKey file, which defines the key for the tiling strategy and identifies different partitioning methods. If not configured, it indicates that the operator does not have a corresponding tiling strategy.
│   │   │   ├── ${op_name}_tiling_data.h                # Optional. TilingData file, which stores configuration information related to the tiling strategy, such as block size and parallelism degree. If not configured, it indicates that the operator does not have a corresponding tiling strategy.
│   │   │   ├── ${op_name}.cpp                          # Kernel entry file, containing the main function and scheduling logic.
│   │   │   └── ${op_name}.h                            # Kernel implementation file, defining the kernel header file, including function declarations, structure definitions, and logic implementations.
│   │   └── tests                                       # Operator test case directory.
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # Optional, UT test cases, develop corresponding test cases according to actual conditions.
│   └── ...
├── docs                                                # Project-related Document Directory
├── examples                                            # End-to-End Operator Development and Invocation示例
│   ├── add_example                                     # AI Core Operator Example Directory
│   │   ├── CMakeLists.txt                              # Operator Compilation Configuration File
│   │   ├── examples                                    # Operator Usage Example Directory
│   │   ├── op_graph                                    # Operator Graph Construction Related Directory
│   │   ├── op_host                                     # Operator Information Library, TilingInferShape相关实现目录
│   │   ├── op_kernel                                   # Operator Kernel Directory
│   │   └── tests                                       # Operator Test Case Directory
│   ├── CMakeLists.txt
│   └── README.md                                       # Project Example Introduction Document
├── scripts                                             # Script directory, containing configuration files related to custom operators and kernel building.
├── tests                                               # Project-level test directory.
├── CMakeLists.txt                                      # Project-level CMakeLists entry.
├── CONTRIBUTING.md                                     # Project contribution guide file.
├── LICENSE                                             # Project open-source license information.
├── OAT.xml                                             # Configuration script for using code repository tools to check whether the license is compliant.
├── README.md                                           # Overall project introduction document.
├── SECURITY.md                                         # Project security statement file.
├── build.sh                                            # Project compilation script.
├── classify_rule.yaml                                  # Component division information.
├── install_deps.sh                                     # Script for installing project dependencies.
├── requirements.txt                                    # Third-party dependency packages of the project.
└── version.info                                        # Project version information
```

