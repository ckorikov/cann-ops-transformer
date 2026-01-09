/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_vector_comm.h
 * \brief 存放各种vector的公共组件
 */

#ifndef COMPRESSOR_VECTOR_COMM_H
#define COMPRESSOR_VECTOR_COMM_H

#include "compressor_comm.h"
namespace Compressor {


struct MatRpeatParam {
    uint32_t row;
    uint32_t col;
    uint32_t dtypeMask;
    uint32_t loopTimes;
    uint32_t axisRemain;
    uint32_t repeatStride;
};

struct RmsNormParam{
    float reciprocal;
    float epsilon;
    uint32_t row;
    uint32_t col;
    float scale;
    uint16_t isScaleEnable;
};

__aicore__ inline void ColumnSum(LocalTensor<float> outputTensor, LocalTensor<float> inputTensor, uint32_t row,
                                 uint32_t col)
{
    if (unlikely(row == 1)) {
        Datacopy(outputTensor, inputTensor, row * col);
        return;
    }
    // 将srcUb的dealRowCount行累加到第一行,每行columnCount各元素
    for (uint32_t mask = MAX_R * 2; mask > 1; mask >>= 1) {
        if (row & mask) {
            Add(outputTensor, inputTensor, inputTensor[mask * col / 2], mask * col / 2);
            PipeBarrier<PIPE_V>();
            if (unlikely(row > mask)) {
                Add(outputTensor, outputTensor, inputTensor[mask * col], (row - mask) * col);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t i = mask >> 2; i > 0; i >>= 1) {
                Add(outputTensor, outputTensor, outputTensor[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            break;
        }
    }
}

__aicore__ inline void ColumnMax(LocalTensor<float> outputTensor, LocalTensor<float> inputTensor, uint32_t row,
                                 uint32_t col)
{
    if (unlikely(row == 1)) {
        Datacopy(outputTensor, inputTensor, row * col);
        return;
    }
    // 将srcUb的dealRowCount行累加到第一行,每行columnCount各元素
    for (uint32_t mask = MAX_R * 2; mask > 1; mask >>= 1) {
        if (row & mask) {
            Max(outputTensor, inputTensor, inputTensor[mask * col / 2], mask * col / 2);
            PipeBarrier<PIPE_V>();
            if (unlikely(row > mask)) {
                Max(outputTensor, outputTensor, inputTensor[mask * col], (row - mask) * col);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t i = mask >> 2; i > 0; i >>= 1) {
                Max(outputTensor, outputTensor, outputTensor[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            break;
        }
    }
}

__aicore__ inline void MatSubVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    uint32_t offset = 0;
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = min(repeatParam.row - row + REPEAT_MAX_NUM, REPEAT_MAX_NUM);
        for (uint32_t i = 0; i < dLoop; i++) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.axisRemain > 0) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.axisRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

__aicore__ inline void MatDivVec(LocalTensor<float> &dstLocal, LocalTensor<float> &src0Local,
                                 LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    uint32_t offset = 0;
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = min(repeatParam.row - row + REPEAT_MAX_NUM, REPEAT_MAX_NUM);
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.axisRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.axisRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

__aicore__ inline void RowSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                              const LocalTensor<float> &shareTmpUb, const MatRpeatParam &repeatParam)
{
    uint32_t blockCount = rowColRepeatParams.loopTimes;
    if (blockCount > 0 && repeatParam.axisRemain > 0) {
        Add(shareTmpUb, srcLocal, srcLocal[blockCount * repeatParam.dtypeMask], repeatParam.axisRemain,
            rowColRepeatParams.row,
            {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        AscendC::PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount >> 1; loopCount > 0; loopCount = blockCount >> 1) {
        blockCount = (blockCount + 1) >> 1;
        for (uint32_t i = 0; i < loopCount; i++) {
            Add(shareTmpUb[i * repeatParam.dtypeMask], srcLocal[i * repeatParam.dtypeMask],
                srcLocal[(i + blockCount) * repeatParam.dtypeMask], repeatParam.dtypeMask, rowColRepeatParams.row,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstLocal, shareTmpUb,
                   (rowColRepeatParams.col < rowColRepeatParams.dtypeMask) ? rowColRepeatParams.col :
                                                                             rowColRepeatParams.dtypeMask,
                   rowColRepeatParams.row, 1, 1, repeatParam.repeatStride);
}

__aicore__ inline void RowDivs(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                               const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    uint32_t offset = 0;
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = min(repeatParam.row - row + REPEAT_MAX_NUM, REPEAT_MAX_NUM);
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.axisRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.axisRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief RowMuls muls by row, 每行的元素乘以相同的元素，该元素需要扩展到一个数据块；
 *        dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
 * @param dstUb 输出tensor [row, columnStride]
 * @param src0Ub 输入tensor [row, columnStride]
 * @param src1Ub 输入tensor [row, FP32_BLOCK_ELEMENT_NUM]
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void RowMuls(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                               const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    uint32_t offset = 0;
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = min(repeatParam.row - row + REPEAT_MAX_NUM, REPEAT_MAX_NUM);
        for (uint32_t i = 0; i < dLoop; i++) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.axisRemain > 0) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.axisRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

} // namespace Compressor
#endif // COMPRESSOR_VECTOR_COMM_H