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
 * \file moe_token_unpermute_with_routing_map_grad.cpp
 * \brief
 */
#ifndef _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_
#define _MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

#define DTYPE_PERMUTED_TOKENS bfloat16_t

#define DTYPE_PROBS bfloat16_t

struct MoeTokenUnpermuteWithRoutingMapGradTilingData {
    int64_t tokensNum = 0;
    int64_t topK = 0;
    int64_t capacity = 0;
    int64_t numExpert = 0;
    int64_t hiddenSize = 0;
    int64_t numOutTokens = 0;
    int64_t formerCoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t tokenNumEachCore = 0;
    int64_t tokenNumTailCore = 0;
    int64_t rowIdMapEachCore = 0;
    int64_t rowIdMapTailCore = 0;

    int64_t hiddenSizeAlign = 0;
    int64_t hiddenSizeLoopTimes = 0;
    int64_t hiddenSizeLoopTimesAlign = 0;
    int64_t hiddenSizeTail = 0;
    int64_t inputReserveNum = 0;
    int64_t indicesReserveNum = 0;
    int64_t indicesReserveNumAlign = 0;
    int64_t numExpertAlign = 0;
    int64_t totalUbSize = 0;
};

inline void InitMoeTokenUnpermuteWithRoutingMapGradTilingData(
    uint8_t* tiling, MoeTokenUnpermuteWithRoutingMapGradTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeTokenUnpermuteWithRoutingMapGradTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)            \
    MoeTokenUnpermuteWithRoutingMapGradTilingData tilingData; \
    InitMoeTokenUnpermuteWithRoutingMapGradTilingData(tilingPointer, &tilingData)
#endif