/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_permute_grad_tiling.cpp
 * \brief
 */

#include "moe_token_permute_grad_tiling.h"

namespace optiling {

static ge::graphStatus Tiling4MoeTokenPermuteGrad(gert::TilingContext *context)
{
    const int *top_k = context->GetAttrs()->GetAttrPointer<int>(0);
    int64_t topk = static_cast<int64_t>(*top_k);
    return TilingCompute(context, topk);
}

static ge::graphStatus TilingPrepareForMoeTokenPermuteGrad(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenPermuteGradCompileInfo {};

IMPL_OP_OPTILING(MoeTokenPermuteGrad)
    .Tiling(Tiling4MoeTokenPermuteGrad)
    .TilingParse<MoeTokenPermuteGradCompileInfo>(TilingPrepareForMoeTokenPermuteGrad);
} // namespace optiling