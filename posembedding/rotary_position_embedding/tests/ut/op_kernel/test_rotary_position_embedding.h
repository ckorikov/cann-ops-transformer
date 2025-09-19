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

#ifndef _ROTARY_POSITION_EMBEDDING_TILING_H_
#define _ROTARY_POSITION_EMBEDDING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

inline void InitRotaryPositionEmbeddingTilingData(uint8_t *tiling, RotaryPositionEmbeddingTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(RotaryPositionEmbeddingTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                                                                     \
    RotaryPositionEmbeddingTilingData tilingData;                                                                      \
    InitRotaryPositionEmbeddingTilingData(tilingPointer, &tilingData)
#endif
