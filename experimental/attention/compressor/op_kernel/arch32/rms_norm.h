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
 * \file rms_norm.h
 * \brief
 */

#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../compressor_comm.h"
#include "../compressor_vector_comm.h"

namespace Compressor {
/**
 * @brief RmsNorm 对矩阵进行rmsnorm
 * @param outLocal 输出tensor [row * col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row * col]
 * @param gammaLocal 系数gamma [1 * col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * sizeof(float) + row *  ALIGN_BLOCK_SIZE]
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数
          col 列数
 */
template <typename GammaType>
__aicore__ inline void RmsNorm(const LocalTensor<float> &outLocal, const LocalTensor<float> &inputLocal,
                               const LocalTensor<GammaType> &gammaLocal, const LocalTensor<uint8_t> &shareTmpUb,
                               const RmsNormParam &rmsNormParams)
{
    uint64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> xSquareLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> xSumLocal = xSquareLocal[cnt];
    Mul(xSquareLocal, inputLocal, inputLocal, cnt);
    AscendC::PipeBarrier<PIPE_V>();
    
    MatRpeatParam repeatParams = {
        rmsNormParams.row;                                // row
        rmsNormParams.col;                                // col
        FP32_REPEAT_ELEMENT_NUM;                          // dtypeMask
        rmsNormParams.col / dtypeMask;                    // loopTimes
        rmsNormParams.col % dtypeMask;                    // axisRemain
        rmsNormParams.col / FP32_BLOCK_ELEMENT_NUM;       // repeatStride
    };

    RowSum(xSumLocal, xSquareLocal, repeatParams)
    AscendC::PipeBarrier<PIPE_V>();


    // Calc: xSum = xSum * reciprocal
    Muls<float>(xSumLocal, xSumLocal, rmsNormParams.row);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSum = xSum + epsilon
    Adds<float>(xSumLocal, xSumLocal, rmsNormParams.row);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSum = sqrt(xSum)
    Sqrt(xSumLocal, xSumLocal, rmsNormParams.row);
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: xSquare[1, 8] = brc(xSum[1,1])
    // TODO:ceil
    Brcb(xSquareLocal, xSumLocal, rmsNormParams.row / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    // Calc: inputLocal = inputLocal / xSquareLocal
    // Div(inputLocal, inputLocal, xSquareLocal, mask, cnt / 64, divParams);

    RowDivs(inputLocal, inputLocal, xSquareLocal, repeatParams);

    AscendC::PipeBarrier<PIPE_V>();

    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, rmsNormParams.col);

    AscendC::PipeBarrier<PIPE_V>();

    RowMuls(outLocal, inputLocal, xSquareLocal, repeatParams);
}
} // namespace Compressor
#endif // MLA_PROLOG_RMS_NORM_H