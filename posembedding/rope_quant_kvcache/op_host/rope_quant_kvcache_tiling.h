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
 * \file rope_quant_kvcache_tiling.h
 * \brief
 */
#ifndef ROPE_QUANT_KVCACHE_TILING_H
#define ROPE_QUANT_KVCACHE_TILING_H
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_templates_registry.h" 
#include "util/math_util.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RopeQuantKvcacheTilingData)
TILING_DATA_FIELD_DEF(uint64_t, qHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, cacheSeqlen);
TILING_DATA_FIELD_DEF(uint64_t, qHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, kHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, vHiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeQuantKvcache, RopeQuantKvcacheTilingData)
struct RopeQuantKvcacheCompileInfo {
};
} // namespace optiling
#endif