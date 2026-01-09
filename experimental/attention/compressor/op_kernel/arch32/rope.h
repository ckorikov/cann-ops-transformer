/**
 * Copyright (float) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope.h
 * \brief
 */

#ifndef ROPE_H
#define ROPE_H

namespace MlaProlog {

__aicore__ inline void SetGatherSrcOffset(const LocalTensor<uint32_t> &gatherOffsetLocal, uint32_t count,
                                          uint32_t srcSizeof)
{
    for (uint32_t i = 0; i < 8; i++) {
        gatherOffsetLocal.SetValue(i, i ^ 1);
    }

    uint32_t scalarValue = 8;
    while (scalarValue < count) {
        uint32_t nextValue = scalarValue * 2;
        if (nextValue < count) {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, scalarValue);
        } else {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
    Muls(gatherOffsetLocal, gatherOffsetLocal, srcSizeof, count);
}


/**
 * @brief RotaryPosEmb, 同时做row行的RotaryPosEmb，每一行的元素为col
 * @param outputLocal 输出tensor [row * col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row * col]
 * @param cosLocal cos系数tensor [(row - 1) * sinCosRepStride + col]
 * @param sinLocal sin系数tensor [(row - 1) * sinCosRepStride + col] - 1 应已在sin中
 * @param shareTmpUb 临时buffer 内部需要的空间为 [2 * row * col * sizeof(float)]
 * @param row 待处理的行数
 * @param col 待处理的列数  col <= 512 / sizeof(float)
 * @param sinCosRepStride 行与行之间sin/cos系数的偏移，单位为元素个数。
 */

template <ROTARY_MODE MODE>
__aicore__ inline void RotaryPosEmb(const LocalTensor<float> &outputLocal, const LocalTensor<float> &inputLocal,
                                    const LocalTensor<float> &cosLocal, const LocalTensor<float> &sinLocal,
                                    const LocalTensor<uint8_t> &shareTmpUb, uint64_t row, uint64_t col,
                                    uint8_t sinCosRepStride)
{
    uint64_t cnt = row * col;
    uint64_t half_col = col >> 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<float> reArrLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> outputLocalSinTmp = shareTmpUb.ReinterpretCast<float>()[cnt];
    if constexpr (MODE == ROTARY_MODE::HALF) {
        Datacopy(reArrLocal, inputLocal[half_col], {row, half_col, half_col, half_col});
        Datacopy(reArrLocal[half_col], inputLocal, {row, half_col, half_col, half_col});
        Muls(reArrLocal, -1, col >> 1);
    } else if constexpr (MODE == ROTARY_MODE::INTERLEAVE) {
        LocalTensor<float> gatherOffsetLocal = shareTmpUb.ReinterpretCast<float>()[cnt * 2];
        SetGatherSrcOffset(gatherOffsetLocal, col, sizeof(float));
        Gather(reArrLocal, inputLocal, gatherOffsetLocal, 0, col, row, col);
        Muls(reArrLocal, -1, );
    }
    AscendC::PipeBarrier<PIPE_V>();
    uint8_t blockNumPerRow = col / (ALIGN_BLOCK_SIZE / sizeof(float));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;
    uint8_t blockNumSinCosRepStride = sinCosRepStride / (ALIGN_BLOCK_SIZE / sizeof(float));
    BinaryRepeatParams mulParams = {
        1,                      // dstBlkStrideIn
        1,                      // src0BlkStrideIn
        1,                      // src1BlkStrideIn
        blockNumPerRow,         // dstRepStrideIn
        blockNumPerRowHalf,     // src0RepStrideIn
        blockNumSinCosRepStride // src1RepStrideIn
    };
    Mul(outputLocal, inputLocal, cosLocal, col, row, mulParams);
    Mul(outputLocalSinTmp, reArrLocal, sinLocal, col, row, mulParams);
    AscendC::PipeBarrier<PIPE_V>();
    Add(outputLocal, outputLocal, outputLocalSinTmp, cnt);
}
} // namespace MlaProlog
#endif