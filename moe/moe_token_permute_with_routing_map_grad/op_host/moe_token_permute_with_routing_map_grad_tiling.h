/**
  * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. reserved.
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
 * \file moe_token_permute_with_routing_map_grad.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H

#include "moe_token_permute_grad_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MoeTokenpermuteWithRoutingMapDropPadTilingData)
TILING_DATA_FIELD_DEF(int64_t, capacity)
TILING_DATA_FIELD_DEF(int64_t, expertNum)
TILING_DATA_FIELD_DEF(int64_t, tokenNum)
TILING_DATA_FIELD_DEF(int64_t, singleCoreLen)
TILING_DATA_FIELD_DEF(int64_t, lastCoreLen)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(
    MoeTokenpermuteWithRoutingMapDropPadTilingDataOp, MoeTokenpermuteWithRoutingMapDropPadTilingData)

REGISTER_TILING_DATA_CLASS(MoeTokenPermuteWithRoutingMapGradUnpermuteTilingDataOp, MoeTokenPermuteWithRoutingMapGradUnpermuteTilingData)

BEGIN_TILING_DATA_DEF(MoeTokenPermuteWithRoutingMapGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MoeTokenPermuteWithRoutingMapGradUnpermuteTilingData, moeTokenPermuteWithRoutingMapGradUnpermuteTilingData)
TILING_DATA_FIELD_DEF_STRUCT(
    MoeTokenpermuteWithRoutingMapDropPadTilingData, moeTokenpermuteWithRoutingMapDropPadTilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MoeTokenPermuteWithRoutingMapGrad, MoeTokenPermuteWithRoutingMapGradTilingData)

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_GRAD_H
