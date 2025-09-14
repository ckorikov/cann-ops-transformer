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
#ifndef TEST_DEQUANT_ROPE_QUANT_KVCACHE_H
#define TEST_DEQUANT_ROPE_QUANT_KVCACHE_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define DTYPE_COS half
#define DTYPE_X half
#define __aicore__

struct DequantRopeQuantKvcacheTilingData {
    int64_t qHeadNum = 8;
    int64_t kvHeadNum = 1;
    int64_t hiddenSize = 128;
    int64_t hiddenSizeFp32Align = 128 * 4;
    int64_t hiddenSizeFp16Align = 128 * 2;
    int64_t hiddenSizeInt8Align = 128;
    int64_t OnceUBMaxS = 15;
    int64_t cacheSeqlen = 1024;
    int64_t seqlen = 1;
    int64_t qHiddenSize = 128 * 8;
    int64_t kHiddenSize = 128;
    int64_t vHiddenSize = 128;
    int64_t realCoreNum = 40;
    int64_t frontCoreNum = 30;
    int64_t blockFactor = 3;
    int64_t tailCoreBlockFactor = 10;
    int64_t hasQuantOffset = 1;
    int64_t ifKVout = 1;
    int64_t isPA = 1;
    int64_t hasBias = 1;
    int64_t hasAS = 1;
};

inline void IDequantRopeQuantKvcacheTilingData(uint8_t* tiling, DequantRopeQuantKvcacheTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(DequantRopeQuantKvcacheTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    DequantRopeQuantKvcacheTilingData tilingData;  \
    IDequantRopeQuantKvcacheTilingData(tilingPointer, &tilingData)
#endif