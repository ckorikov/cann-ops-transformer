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
/**
 * @brief ColumnSoftMax 对矩阵按列进行SoftMax
 * @param outLocal 输出tensor [row * col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row * col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [floor(row / 2) * col * sizeof(float)]
 * @param row 行数
 * @param col 列数
 */
__aicore__ inline void ColumnSoftMax(const LocalTensor<float> &outLocal, const LocalTensor<float> &inputLocal,
                                     const LocalTensor<float> &shareTmpUb, const uint32_t row, const uint32_t col)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = col / dtypeMask;
    uint32_t dRemain = col % dtypeMask;
    uint32_t repeatStride = col / FP32_BLOCK_ELEMENT_NUM;
    ColumnMax(shareTmpUb, inputLocal, row, col);
    MatSubsVec(outLocal, inputLocal, shareTmpUb, {row, col, dtypeMask, dLoop, dRemain, repeatStride});
    PipeBarrier<PIPE_V>();
    Exp(outLocal, outLocal, row * col);
    PipeBarrier<PIPE_V>();
    ColumnSum(shareTmpUb, outLocal, row, col);
    MatDivsVec(outLocal, outLocal, shareTmpUb, {row, col, dtypeMask, dLoop, dRemain, repeatStride});
}

} // namespace Compressor

#endif
