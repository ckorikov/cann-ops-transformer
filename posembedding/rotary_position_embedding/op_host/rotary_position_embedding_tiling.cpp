/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
 * \file rotary_position_embedding.cc
 * \brief
 */
#include "rotary_position_embedding_tiling.h"
#include "rope_rotate_half_tiling.h"
#include "rope_interleaved_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
constexpr uint32_t MODE_ATTR_IDX = 0;

ge::graphStatus RotaryPosEmbeddingMembaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        socVersion_ = ascendcPlatform.GetSocVersion();
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const RotaryPositionEmbeddingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus RotaryPosEmbeddingMembaseTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const uint32_t inputMode = *(attrs->GetAttrPointer<uint32_t>(MODE_ATTR_IDX));
    OP_LOGI(context_->GetNodeName(), "[mode]: %d", inputMode);
    inputMode_ = inputMode;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RotaryPositionEmbedding(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForRotaryPositionEmbedding(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("RotaryPositionEmbedding", RopeRotateHalfTilingClass, 50000);
REGISTER_TILING_TEMPLATE("RotaryPositionEmbedding", RopeInterLeavedTilingClass, 60000);

IMPL_OP_OPTILING(RotaryPositionEmbedding)
    .Tiling(Tiling4RotaryPositionEmbedding)
    .TilingParse<RotaryPositionEmbeddingCompileInfo>(TilingPrepareForRotaryPositionEmbedding);
} // namespace optiling
