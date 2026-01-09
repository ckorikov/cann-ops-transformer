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
 * \file soft_max.h
 * \brief
 */

#ifndef SOFT_MAX_H
#define SOFT_MAX_H

#include "../compressor_comm.h"
#include "../compressor_vector_comm.h"

namespace Compressor {
    __aicore__ inline void ColumnSoftMax(LocalTensor<float> outputTensor, LocalTensor<float> inputTensor, LocalTensor<float> tempTensor,
                                        uint32_t dealRowCount, uint32_t columnCount)
    {
        uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t dLoop = columnCount / dtypeMask;
        uint32_t dRemain = columnCount % dtypeMask;
        uint32_t repeatStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        ColumnMax(tempTensor, inputTensor, dealRowCount, columnCount);
        MatSubsVec(inputTensor, inputTensor, tempTensor, dealRowCount, columnCount, dtypeMask, dLoop, dRemain, repeatStride);
        PipeBarrier<PIPE_V>();
        Exp(inputTensor, inputTensor, dealRowCount * columnCount);
        PipeBarrier<PIPE_V>();
        ColumnSum(tempTensor, inputTensor, dealRowCount, columnCount);
        MatDivsVec(inputTensor, inputTensor, tempTensor, dealRowCount, columnCount, dtypeMask, dLoop, dRemain, repeatStride);
    }
}

#endif


