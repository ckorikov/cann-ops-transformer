# 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

算子接口列表如下：

|    接口名   |      说明     |
|-----------|------------|
|[aclnnGroupedMatmulSwigluQuant](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|实现融合GroupedMatmul 、dquant、swiglu和quant运算。|
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|实现融合GroupedMatmul 、dquant、swiglu和quant运，是aclnnGroupedMatmulSwigluQuant接口的weightNZ特化版本。|
|[aclnnNsaCompressAttentionInfer](../../attention/nsa_compress_attention_infer/docs/aclnnNsaCompressAttentionInfer.md)|实现Native Sparse Attention推理过程中，Compress Attention的计算。|
|[aclnnNsaCompressWithCache](../../attention/nsa_compress_with_cache/docs/aclnnNsaCompressWithCache.md)|实现Native-Sparse-Attention推理阶段的KV压缩。|
|[aclnnPromptFlashAttentionV3](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)|实现全量推理场景的FlashAttention算子，支持sparse优化、actualSeqLengthsKv优化、int8量化功能、innerPrecise参数|
|[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)|在全量推理场景的FlashAttention算子的基础上实现增量推理|
|[aclnnNsaSelectedAttentionInfer](../../attention/nsa_select_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md)|实现Native Sparse Attention推理过程中，Selected Attention的计算。|
|[aclnnFusedInferAttentionScoreV4](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)|适配decode & prefill场景的FlashAttention算子|
