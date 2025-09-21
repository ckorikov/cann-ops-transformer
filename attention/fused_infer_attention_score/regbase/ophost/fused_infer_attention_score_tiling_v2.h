/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_V2_H_
#include "../../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "../../../prompt_flash_attention/regbase/ophost/prompt_flash_attention_tiling_v2.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)
ge::graphStatus TilingFusedInferAttentionScoreV2(gert::TilingContext *context);
} // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_V2_H_