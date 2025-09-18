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
 * \file moe_gating_top_k_softmax_tiling.cpp
 * \brief
 */
#include "log/log.h"
// #include "error_log.h"
// #include "error_util.h"
// #include "op_tiling_util.h"
// #include "runtime2_util.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "moe_gating_top_k_softmax_tiling.h"
// #include "external/exe_graph/runtime/shape.h"
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
namespace optiling {

static ge::graphStatus TilingMoeGatingTopKSoftmax(gert::TilingContext* context)
{
    // 初始化算子Tiling类
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MoeGatingTopKSoftmax(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeGatingTopKSoftmax)
    .Tiling(TilingMoeGatingTopKSoftmax)
    .TilingParse<MoeGatingTopKSoftmaxCompileInfo>(TilingPrepare4MoeGatingTopKSoftmax);
} // namespace optiling