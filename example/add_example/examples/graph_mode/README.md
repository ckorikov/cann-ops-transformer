## 概述

通过AddExample样例算子原型构建图。

## 目录结构介绍

```
├── graph_mode
│   ├── CMakeLists.txt                          // 编译规则文件
│   ├── test_geir_add_example.cpp               // 单算子调用应用的入口
│   └── run.sh                                  // 编译运行算子的脚本
```

## 代码实现介绍

使用REG_OP宏将算子原型注册成功后，会自动生成对应的衍生接口（[《参见原型定义衍生接口》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/basicdataapi/atlasopapi_07_00528.html)），用户可以通过这些接口在Graph中定义算子，然后创建一个Graph实例，并在Graph中设置输入算子、输出算子，从而完成Graph构建。

通过算子原型构图方法参见[《通过算子原型构建Graph》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/graph/graphdevg/atlasag_25_0001.html)中“构建Graph > 通过算子原型构建Graph”章节

## 运行样例算子
**请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**
  
- 进入样例代码所在路径
  
  ```bash
  cd ${git_clone_path}/ops-math-dev/example/add_example/examples/graph_mode/
  ```
  
- 样例执行
    
  样例执行过程中会自动生成测试数据，然后编译与运行图，最后打印运行结果。

  ```bash
  bash run.sh
  ```

## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/08/19 | 新增本readme |
