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
 * \file moe_token_unpermute_with_routing_map_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteWithRoutingMapGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokensNum); // tokens 轴大小
TILING_DATA_FIELD_DEF(int64_t, topK);      // topK 轴大小
TILING_DATA_FIELD_DEF(int64_t, capacity);
TILING_DATA_FIELD_DEF(int64_t, numExpert);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);   // hiddenSize 轴大小
TILING_DATA_FIELD_DEF(int64_t, numOutTokens); // input的第0维输入

TILING_DATA_FIELD_DEF(int64_t, formerCoreNum);    // 前core_num - tailCoreNum轴
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);      // 尾轴数
TILING_DATA_FIELD_DEF(int64_t, tokenNumEachCore); // 前core_num - tailCoreNum轴计算的token_num数据量
TILING_DATA_FIELD_DEF(int64_t, tokenNumTailCore); // 后tailCoreNum轴计算的token_num数据量
TILING_DATA_FIELD_DEF(int64_t, rowIdMapEachCore); // 前core_num - tailCoreNum轴计算的sorted_indices数据量
TILING_DATA_FIELD_DEF(int64_t, rowIdMapTailCore); // 后tailCoreNum轴计算的sorted_indices数据量

TILING_DATA_FIELD_DEF(int64_t, hiddenSizeAlign);     // hiddenSize对齐
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeLoopTimes); // hiddenSize循环次数
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeLoopTimesAlign);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeTail);         // hiddenSize尾块
TILING_DATA_FIELD_DEF(int64_t, inputReserveNum);        // input一次搬入的数量
TILING_DATA_FIELD_DEF(int64_t, indicesReserveNum);      // sorted_indices一次搬入的数量
TILING_DATA_FIELD_DEF(int64_t, indicesReserveNumAlign); // sorted_indices一次搬入的数量对齐
TILING_DATA_FIELD_DEF(int64_t, numExpertAlign);

TILING_DATA_FIELD_DEF(int64_t, totalUbSize); // ub空间总大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTokenUnpermuteWithRoutingMapGrad, MoeTokenUnpermuteWithRoutingMapGradTilingData)
} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H
