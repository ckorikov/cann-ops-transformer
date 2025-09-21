/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_def.h
 * \brief
 */

#pragma once
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(MLATensorTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, t1);
TILING_DATA_FIELD_DEF(int64_t, t2);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, qSize);
TILING_DATA_FIELD_DEF(int64_t, kvSize);
TILING_DATA_FIELD_DEF(int64_t, sfmgSize);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(int64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, dkWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, dvWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, sfmgWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, mm1WorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, mm2WorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, pWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, dsWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint8_t, tndSoftmaxIn);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MLATensorTilingDataOp, MLATensorTilingData)

BEGIN_TILING_DATA_DEF(FlashAttentionGradMlaTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MLATensorTilingData, mlaTensorTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_11011000000123456789, FlashAttentionGradMlaTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_11011000000123456788, FlashAttentionGradMlaTilingData)
} // namespace optiling
