/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_gating_top_k_softmax_v2_tiling.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
namespace optiling {

static ge::graphStatus TilingMoeGatingTopKSoftmaxV2(gert::TilingContext* context)
{
    // 初始化算子Tiling类
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MoeGatingTopKSoftmaxV2(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeGatingTopKSoftmaxV2)
    .Tiling(TilingMoeGatingTopKSoftmaxV2)
    .TilingParse<MoeGatingTopKSoftmaxV2CompileInfo>(TilingPrepare4MoeGatingTopKSoftmaxV2);
} // namespace optiling