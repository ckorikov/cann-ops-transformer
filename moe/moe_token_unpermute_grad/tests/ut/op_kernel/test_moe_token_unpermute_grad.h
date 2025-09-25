/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_token_unpermute_grad.h
 * \brief
 */
#ifndef _MOE_TOKEN_UNPERMUTE_GRAD_TILING_H_
#define _MOE_TOKEN_UNPERMUTE_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct MoeTokenUnpermuteGradTilingData {
    int64_t tokensNum = 0;
    int64_t topK = 0;
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
    int64_t hiddenSizeTail = 0;
    int64_t inputReserveNum = 0;
    int64_t indicesReserveNum = 0;
    int64_t indicesReserveNumAlign = 0;
    int64_t totalUbSize = 0;
};

inline void InitMoeTokenUnpermuteGradTilingData(uint8_t* tiling, MoeTokenUnpermuteGradTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeTokenUnpermuteGradTilingData));
}
#define DTYPE_PERMUTED_TOKENS bfloat16_t
#define DTYPE_PROBS float
#define GET_TILING_DATA(tilingData, tilingPointer) \
    MoeTokenUnpermuteGradTilingData tilingData;    \
    InitMoeTokenUnpermuteGradTilingData(tilingPointer, &tilingData)
#endif