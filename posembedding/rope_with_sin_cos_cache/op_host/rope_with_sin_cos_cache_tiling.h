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
 * \file rope_with_sin_cos_cache.h
 * \brief
 */

#ifndef ROPE_WITH_SIN_COS_CACHE_H
#define ROPE_WITH_SIN_COS_CACHE_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopeWithSinCosCacheTilingData)
TILING_DATA_FIELD_DEF(uint64_t, core_num_use);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens);
TILING_DATA_FIELD_DEF(uint64_t, num_q_heads);
TILING_DATA_FIELD_DEF(uint64_t, num_kv_heads);
TILING_DATA_FIELD_DEF(uint64_t, head_size);
TILING_DATA_FIELD_DEF(uint64_t, rotary_dim);
TILING_DATA_FIELD_DEF(uint64_t, mrope_section0);
TILING_DATA_FIELD_DEF(uint64_t, mrope_section1);
TILING_DATA_FIELD_DEF(uint64_t, mrope_section2);
TILING_DATA_FIELD_DEF(uint64_t, q_leading_dimension);
TILING_DATA_FIELD_DEF(uint64_t, k_leading_dimension);
TILING_DATA_FIELD_DEF(uint64_t, isNeoxStyle);
TILING_DATA_FIELD_DEF(uint64_t, front_core);
TILING_DATA_FIELD_DEF(uint64_t, tail_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_each_front_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_each_tail_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_front_core_each_loop);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_tail_core_each_loop);
TILING_DATA_FIELD_DEF(uint64_t, loop_time_each_front_core);
TILING_DATA_FIELD_DEF(uint64_t, loop_time_each_tail_core);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_front_core_last_loop);
TILING_DATA_FIELD_DEF(uint64_t, num_tokens_tail_core_last_loop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeWithSinCosCache, RopeWithSinCosCacheTilingData)
} // namespace optiling

#endif