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
    uint32_t colRemain;
    uint32_t repeatStride;
};

struct RmsNormParam {
    float reciprocal;
    float epsilon;
    uint32_t row;
    uint32_t col;
    float scale;
    uint16_t isScaleEnable;
};

/**
 * @brief ColumnSum 对矩阵按列进行求和
 * @param dstLocal 输出tensor [1, col]，支持和shareTmpUb是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [floor(row / 2) * col * sizeof(float)]
 * @param row 行数
 * @param col 列数
 */
__aicore__ inline void ColumnSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                     const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    // 行数为1时，直接将srcLocal复制到dstLocal
    if (unlikely(row == 1)) {
        Datacopy(dstLocal, srcLocal, row * col);
        return;
    }
    for (uint32_t mask = MAX_R * 2; mask > 1; mask >>= 1) {
        if (row & mask) {
            // 将输入对半求和后放进临时空间
            Add(shareTmpUb, srcLocal, srcLocal[mask * col / 2], mask * col / 2);
            PipeBarrier<PIPE_V>();
            // 将余量加到前一半上
            if (unlikely(row > mask)) {
                Add(shareTmpUb, shareTmpUb, srcLocal[mask * col], (row - mask) * col);
                PipeBarrier<PIPE_V>();
            }
            // 每次将后一半行加到前一半上
            for (uint32_t i = mask >> 2; i > 1; i >>= 1) {
                Add(shareTmpUb, shareTmpUb, shareTmpUb[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            Add(dstLocal, shareTmpUb, shareTmpUb[i * col], i * col);
            PipeBarrier<PIPE_V>();
            break;
        }
    }
}

/**
 * @brief ColumnMax 对矩阵按列进行求最大值
 * @param dstLocal 输出tensor [1, col]，支持和shareTmpUb是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [floor(row / 2) * col * sizeof(float)]
 * @param row 行数
 * @param col 列数
 */
__aicore__ inline void ColumnSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                     const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    // 行数为1时，直接将srcLocal复制到dstLocal
    if (unlikely(row == 1)) {
        Datacopy(dstLocal, srcLocal, row * col);
        return;
    }
    for (uint32_t mask = MAX_R * 2; mask > 1; mask >>= 1) {
        if (row & mask) {
            // 将输入对半求最大值后放进临时空间
            Max(shareTmpUb, srcLocal, srcLocal[mask * col / 2], mask * col / 2);
            PipeBarrier<PIPE_V>();
            // 将余量和前一半求最大值后加到前一半上
            if (unlikely(row > mask)) {
                Max(shareTmpUb, shareTmpUb, srcLocal[mask * col], (row - mask) * col);
                PipeBarrier<PIPE_V>();
            }
            // 每次将后一半行和前一半最大值后加到前一半上
            for (uint32_t i = mask >> 2; i > 1; i >>= 1) {
                Max(shareTmpUb, shareTmpUb, shareTmpUb[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            Max(dstLocal, shareTmpUb, shareTmpUb[i * col], i * col);
            PipeBarrier<PIPE_V>();
            break;
        }
    }
}


/**
 * @brief MatSubVec 矩阵逐行减向量
 * @param dstLocal 输出tensor [row, col]
 * @param src0Local 输入tensor [row, col]
 * @param src1Local 输入tensor [1, col]
 * @param repeatParam 描述待处理数据的排布，包括
            row 行数
            col 列数
            dtypeMask 一次迭代参与计算元素数
            loopTimes 循环次数
            colRemain 剩余列数
            repeatStride 循环步长（内存中实际列长度）
 */
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
        if (repeatParam.colRemain > 0) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief MatDivVec 矩阵逐行除以向量
 * @param dstLocal 输出tensor [row, col]
 * @param src0Local 输入tensor [row, col]
 * @param src1Local 输入tensor [1, col]
 * @param repeatParam 描述待处理数据的排布，包括
            row 行数
            col 列数
            dtypeMask 一次迭代参与计算元素数
            loopTimes 循环次数
            colRemain 剩余列数
            repeatStride 循环步长（内存中实际列长度）
 */
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
        if (repeatParam.colRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief MatMulVec 矩阵逐行乘以向量
 * @param dstLocal 输出tensor [row, col]
 * @param src0Local 输入tensor [row, col]
 * @param src1Local 输入tensor [1, col]
 * @param repeatParam 描述待处理数据的排布，包括
            row 行数
            col 列数
            dtypeMask 一次迭代参与计算元素数
            loopTimes 循环次数
            colRemain 剩余列数
            repeatStride 循环步长（内存中实际列长度）
 */
__aicore__ inline void MatMulVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    uint32_t offset = 0;
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = min(repeatParam.row - row + REPEAT_MAX_NUM, REPEAT_MAX_NUM);
        for (uint32_t i = 0; i < dLoop; i++) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief RowSum 矩阵对每行求和
 * @param dstLocal 输出tensor [1, row]
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row, col]，支持和srcLocal是同一块空间
 * @param repeatParam 描述待处理数据的排布，包括
            row 行数
            col 列数
            dtypeMask 一次迭代参与计算元素数
            loopTimes 循环次数
            colRemain 剩余列数
            repeatStride 循环步长（内存中实际列长度）
 */
__aicore__ inline void RowSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                              const LocalTensor<float> &shareTmpUb, const MatRpeatParam &repeatParam)
{
    uint32_t blockCount = rowColRepeatParams.loopTimes;
    if (blockCount > 0 && repeatParam.colRemain > 0) {
        Add(shareTmpUb, srcLocal, srcLocal[blockCount * repeatParam.dtypeMask], repeatParam.colRemain,
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

/**
 * @brief RowDivs 矩阵每行除以对应元素
 * @param dstLocal 输出tensor [row, col]
 * @param src0Local 输入tensor [row, col]
 * @param src1Local 输入tensor [row, 1]，需要扩展到一个datablock中(实际内存需要为[row, col])
 * @param repeatParam 描述待处理数据的排布，包括
            row 行数
            col 列数
            dtypeMask 一次迭代参与计算元素数
            loopTimes 循环次数
            colRemain 剩余列数
            repeatStride 循环步长（内存中实际列长度）
 */
__aicore__ inline void RowDivs(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                               const LocalTensor<float> &src1Local, const MatRpeatParam &repeatParam)
{
    uint32_t offset = 0;
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = min(repeatParam.row - row + REPEAT_MAX_NUM, REPEAT_MAX_NUM);
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        }
    }
}


/**
 * @brief RowMuls 矩阵每行乘以相同元素
 * @param dstLocal 输出tensor [row, col]
 * @param src0Local 输入tensor [row, col]
 * @param src1Local 输入tensor [row, 1]，需要扩展到一个datablock中(实际内存需要为[row, col])
 * @param repeatParam 描述待处理数据的排布，包括
            row 行数
            col 列数
            dtypeMask 一次迭代参与计算元素数
            loopTimes 循环次数
            colRemain 剩余列数
            repeatStride 循环步长（内存中实际列长度）
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
        if (repeatParam.colRemain > 0) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

} // namespace Compressor
#endif // COMPRESSOR_VECTOR_COMM_H