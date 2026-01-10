/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor.cpp
 * \brief
 */

#include "compressor_kernel.h"
 
using namespace Compressor;

template<uint8_t X_LAYOUT, int8_t X_DTYPE, int8_t COFF, bool ROTARY_MODE>
__global__ __aicore__ void compressor(
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *kvState,
    __gm__ uint8_t *scoreState,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *kvStateOut,
    __gm__ uint8_t *scoreStateOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorTilingData, tilingDataIn, tiling);
    const optiling::CompressorTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    CompressorKernel op(&pipe, tilingData);
    op.Init(x,
            wKv,
            wGate,
            kvState,
            scoreState,
            ape,
            normWeight,
            ropeSin,
            ropeCos,
            blockTable,
            cuSeqlens,
            seqUsed,
            startPos,
            cmpKvOut,
            kvStateOut,
            scoreStateOut,
            workspace);
    op.Process();

}