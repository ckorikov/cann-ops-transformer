/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_common.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_attn_sharedkv_metadata.h"

using namespace AscendC;
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SAS_RUN_MODE {
    SWA_MODE = 0,
    SCFA_MODE = 1,
    CFA_MODE = 2,
};

enum class SAS_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

enum class SAS_KV_LAYOUT {
    TND = 0,
    PA_ND = 1
};

enum class SASTemplateMode {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

enum class QUANT_MODE {
    PER_CHANNEL = 0,  // GQA支持
    PER_TOKEN_HEAD = 1, // GQA支持
    PER_TILE = 2,   // MLA支持
};

enum class ATTENTION_MODE {
    GQA_MHA = 0,  // QKV headDim相等
    MLA_NATIVE = 1, // Dn=128, Dr=64
    MLA_ABSORB = 2,   // Dn=512, Dr=64
};

enum class QUANT_SCALE_REPO_MODE {
    SEPARATE = 0,  // 分开存储
    COMBINE = 1, // 合并存储，量化模式是PER_TOKEN_HEAD/PER_TILE时支持COMBINE模式，参数顺序为：Nope+Rope+DequantScale
};

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
	  SAS_LAYOUT LAYOUT_T = SAS_LAYOUT::BSND, SAS_LAYOUT KV_LAYOUT_T = SAS_LAYOUT::PA_ND, 
      typename... Args>
struct SASType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SAS_LAYOUT layout = LAYOUT_T;
    static constexpr SAS_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SAS_LAYOUT::PA_ND);
};

// ================================Util functions==================================
template <typename T> __aicore__ inline T SASAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}
#endif // KVQUANT_SPARSE_FLASH_ATTENTION_COMMON_H