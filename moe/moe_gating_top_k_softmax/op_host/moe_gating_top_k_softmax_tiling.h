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
 * \file moe_gating_top_k_softmax_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
enum MoeGatingTopKSoftmaxTilingKey
{
    MOE_GATING_SOFTMAX_FLOAT = 0,
    MOE_GATING_SOFTMAX_FLOAT_DOUBLE_BUFFER = 1,
    MOE_GATING_SOFTMAX_FLOAT16 = 2,
    MOE_GATING_SOFTMAX_FLOAT16_DOUBLE_BUFFER = 3,
    MOE_GATING_SOFTMAX_BF16 = 4,
    MOE_GATING_SOFTMAX_BF16_DOUBLE_BUFFER = 5,
    MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT = 6,
    MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT16 = 7,
    MOE_GATING_SOFTMAX_K_FULL_LOAD_BF16 = 8,
    MOE_GATING_SOFTMAX_PERF_FLOAT_COL_SMALLER_THAN_8 = 9,
    MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_SMALLER_THAN_8 = 10,
    MOE_GATING_SOFTMAX_PERF_BF16_COL_SMALLER_THAN_8 = 11,
    MOE_GATING_SOFTMAX_PERF_FLOAT_COL_FROM_8_TO_64 = 12,
    MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_FROM_8_TO_64 = 13,
    MOE_GATING_SOFTMAX_PERF_BF16_COL_FROM_8_TO_64 = 14,
    MOE_GATING_SOFTMAX_PERF_FLOAT_COL_BIGGER_THAN_64 = 15,
    MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_BIGGER_THAN_64 = 16,
    MOE_GATING_SOFTMAX_PERF_BF16_COL_BIGGER_THAN_64 = 17
};

struct MoeGatingTopKSoftmaxCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_H_
