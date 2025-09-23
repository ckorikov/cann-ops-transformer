# 算子清单

> - 算子目录：每个目录承载每个算子所有的交付件，包括代码实现、example、文档等。
> - 算子IR（Intermediate Representation）：表示算子原型，描述了算子输入、输出、属性等信息，包括数据类型、shape、数据格式等。算子清单中有部分算子定义了IR，表明可通过IR构图方式调用算子。

算子清单如下：（补充中）

| 算子分类 | 算子目录                                                                          | 算子IR                                                                                                      | 说明                                                                                       |
| -------- |-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| attention   | [nsa_compress_attention_infer](../../attention/nsa_compress_attention_infer)                                               | -                                                                |  实现Native Sparse Attention推理过程中，Compress Attention的计算。 |
| attention   | [nsa_compress_with_cache](../../attention/nsa_compress_with_cache)                                               | -                                                                |  实现Native-Sparse-Attention推理阶段的KV压缩。 |



