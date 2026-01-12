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

#include "../compressor_comm.h"
#include "../compressor_vector_comm.h"

namespace Compressor {

template <typename T>
__aicore__ inline void SetGatherSrcOffset(const LocalTensor<int32_t> &gatherOffsetLocal, uint32_t count)
{
    for (uint32_t i = 0; i < 8; i++) {
        gatherOffsetLocal.SetValue(i, i ^ 1);
    }
    int32_t scalarValue = 8;
    while (scalarValue < count) {
        int32_t nextValue = scalarValue * 2;
        PipeBarrier<PIPE_V>();
        if (nextValue < count) {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, scalarValue);
        } else {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
    PipeBarrier<PIPE_V>();
    Muls(gatherOffsetLocal, gatherOffsetLocal, static_cast<int32_t>(sizeof(T)), count);
}


/**
 * @brief RotaryPosEmb, 同时做row行的RotaryPosEmb，每一行的元素为col
 * @param outputLocal 输出tensor [row, col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row, col]
 * @param cosLocal cos系数tensor [row, col]
 * @param sinLocal sin系数tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [2 * row * col * sizeof(float)]
 * @param row 待处理的行数
 * @param col 待处理的列数
 * @param sinCosRepStride 行与行之间sin/cos系数的偏移，单位为元素个数。
 */

template <ROTARY_MODE MODE>
__aicore__ inline void RotaryPosEmb(const LocalTensor<float> &outputLocal, const LocalTensor<float> &inputLocal,
                                    const LocalTensor<float> &cosLocal, const LocalTensor<float> &sinLocal,
                                    const LocalTensor<float> &shareTmpUb,
                                    const LocalTensor<uint32_t> &gatherOffsetcastLocal, uint64_t row, uint64_t col)
{
    uint64_t cnt = row * col;
    uint64_t half_col = col >> 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<float> reArrLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> outputLocalSinTmp = shareTmpUb.ReinterpretCast<float>()[cnt];
    if constexpr (MODE == ROTARY_MODE::HALF) {
        DataCopy(reArrLocal, inputLocal[half_col],
                 {static_cast<uint16_t>(row), static_cast<uint16_t>(CeilDivT(half_col, BYTE_BLOCK)),
                  static_cast<uint16_t>(CeilDivT(half_col, BYTE_BLOCK)),
                  static_cast<uint16_t>(CeilDivT(half_col, BYTE_BLOCK))});
        DataCopy(reArrLocal[half_col], inputLocal,
                 {static_cast<uint16_t>(row), static_cast<uint16_t>(CeilDivT(half_col, BYTE_BLOCK)),
                  static_cast<uint16_t>(CeilDivT(half_col, BYTE_BLOCK)),
                  static_cast<uint16_t>(CeilDivT(half_col, BYTE_BLOCK))});
        PipeBarrier<PIPE_V>();
        Muls(reArrLocal, reArrLocal, -1.0f, half_col);
    } else if constexpr (MODE == ROTARY_MODE::INTERLEAVE) {
        for (uint32_t i = 0; i < row; i++) {
            Gather(reArrLocal[i * col], inputLocal[i * col], gatherOffsetcastLocal, 0, col);
        }
        PipeBarrier<PIPE_V>();
        
        uint64_t mask[1] = {0xAAAAAAAAAAAAAAAA};
        Muls(reArrLocal, reArrLocal, -1.0f, mask, row, {1, 1, FP32_BLOCK_ELEMENT_NUM, FP32_BLOCK_ELEMENT_NUM});
    }
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams mulParams = {
        1,                                                  // dstBlkStrideIn
        1,                                                  // src0BlkStrideIn
        1,                                                  // src1BlkStrideIn
        static_cast<uint8_t>(col / FP32_BLOCK_ELEMENT_NUM), // dstRepStrideIn
        static_cast<uint8_t>(col / FP32_BLOCK_ELEMENT_NUM), // src0RepStrideIn
        static_cast<uint8_t>(col / FP32_BLOCK_ELEMENT_NUM)  // src1RepStrideIn
    };
    Mul(outputLocal, inputLocal, cosLocal, col, row, mulParams);
    Mul(outputLocalSinTmp, reArrLocal, sinLocal, col, row, mulParams);
    PipeBarrier<PIPE_V>();
    Add(outputLocal, outputLocal, outputLocalSinTmp, cnt);
}
} // namespace Compressor
#endif