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
 * \file compressor_kernel.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_H
#define COMPRESSOR_KERNEL_H

#include "compressor_comm.h"
#include "compressor_vector_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"

namespace Compressor {

class CompressorKernel {
public:
    __aicore__ inline CompressorKernel(TPipe* pipe, const optiling::CompressorTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
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
        __gm__ uint8_t *hadamard,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut,
        __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    const optiling::CompressorTilingData* __restrict tilingData_;
};

__aicore__ inline void CompressorKernel::Init(
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
        __gm__ uint8_t *hadamard,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut,
        __gm__ uint8_t *workspace) {
    AscendC::printf("CompressorKernel::Init!!!!!\n");
}

__aicore__ inline void CompressorKernel::Process() {
    AscendC::printf("CompressorKernel::Process!!!!!\n");
}


} // namespace Compressor

#endif // COMPRESSOR_KERNEL_H