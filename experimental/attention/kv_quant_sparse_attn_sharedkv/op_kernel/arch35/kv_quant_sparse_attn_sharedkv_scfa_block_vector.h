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
 * \file flash_attention_score_block_vec_base_scfa.h
 * \brief
 */
 // TODO 修改
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_SCFA_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_SCFA_H_
#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h" 

// #include "vf/vf_mul_sel_softmaxflashv2_cast_nz_scfa.h"
// #include "vf/vf_flashupdate_new_scfa.h"

using namespace AscendC;
// using namespace SCFaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF
class SCFABlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    // static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    // static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    // static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    // static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    // static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec() {};
protected:
};

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    // static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    // static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    // TODO 是否需要补充其他函数
    __aicore__ inline SCFABlockVecDummy() {};
};
}
#endif //FLASH_ATTENTION_SCORE_BLOCK_VEC_SCFA_H_

