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
#ifndef APPLY_RAOTRY_POS_EMB_H
#define APPLY_RAOTRY_POS_EMB_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct ApplyRotaryPosEmbTilingData {
    int64_t useCoreNum = 0;
    int64_t lastDim = 0;
    int64_t halfNum = 0;
    int64_t preCBatchB = 0;
    int64_t preCBatchL = 0;
    int64_t lastCBatchL = 0;
    int64_t comBatchBB = 0;
    int64_t comBatchBBL = 0;
    int64_t comBatchBLL = 0;
    int64_t comBatchLLL = 0;
    int64_t qPart1Ub = 0;
    int64_t q2q1Part1Ub = 0;
    int64_t cosPart1Ub = 0;
    int64_t sin1UbSize = 0;
    int64_t preCLTimes = 0;
    int64_t lastCLTimes = 0;
    int64_t preCBBTimes = 0;
    int64_t preCBLTimes = 0;
    int64_t preCLLTimes = 0;
    int64_t qCoreOffset = 0;
    int64_t kCoreOffset = 0;
    int64_t cosCoreOffset = 0;
    int64_t qcNum = 0;
    int64_t kcNum = 0;
    int64_t coscNum = 0;
    int64_t qcdNum = 0;
    int64_t kcdNum = 0;
    int64_t coscdNum = 0;
    int64_t qkcNum = 0;
    int64_t mulNum = 0;
    int64_t qcdHalfNum = 0;
    int64_t dstRepSBr = 0;
    int64_t blockLenQ = 0;
    int64_t srcStrideK = 0;
    int64_t blockLenq2q1 = 0;
    int64_t mask = 0;
    int64_t tilingKey = 0;
};
#define DTYPE_QUERY half
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

inline void InitApplyRotaryPosEmbTilingData(uint8_t* tiling, ApplyRotaryPosEmbTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ApplyRotaryPosEmbTilingData));
}

#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    ApplyRotaryPosEmbTilingData tiling_data;     \
    InitApplyRotaryPosEmbTilingData(tiling_arg, &tiling_data)

#endif
