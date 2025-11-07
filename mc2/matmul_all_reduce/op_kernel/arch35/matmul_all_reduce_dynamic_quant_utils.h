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
 * \file matmul_all_reduce_dynamic_quant_utils.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_DYNAMIC_QUANT_UTILS_H
#define MATMUL_ALL_REDUCE_DYNAMIC_QUANT_UTILS_H

#include "kernel_operator.h"

namespace AscendC {

constexpr uint32_t TILELEN = 128;
constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint32_t UB_DATABLOCK = 32;
constexpr uint32_t COMPARE_ALIGN_LEN = 256;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;

template <class T>
__aicore__ inline void DynamicQuant(uint32_t curScaleCnt, uint32_t padCalCnt, LocalTensor<float> tilesLocal,
                                    LocalTensor<T> curOutTiles, LocalTensor<float> curScale,
                                    LocalTensor<float> tempWorkTiles, LocalTensor<float> tempScale,
                                    LocalTensor<uint8_t> tempMskBuf)
{
    float xTypeMax = std::is_same<T, fp8_e4m3fn_t>::value ? FP8_E4M3_MAX_VALUE : FP8_E5M2_MAX_VALUE;
    const uint32_t broadCastDst[BROADCAST_DIM] = {curScaleCnt, static_cast<uint32_t>(TILELEN)};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curScaleCnt, 1};
    uint32_t compareCnt = (Ceil(curScaleCnt * sizeof(float), COMPARE_ALIGN_LEN) * COMPARE_ALIGN_LEN) / sizeof(float);
    Abs(tempWorkTiles, tilesLocal, padCalCnt);
    Duplicate(tempScale, 0.0f, padCalCnt);
    PipeBarrier<PIPE_V>();
    ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(tempScale, tempWorkTiles, broadCastDst, false);
    PipeBarrier<PIPE_V>();
    CompareScalar(tempMskBuf, tempScale, 0.0f, AscendC::CMPMODE::NE, compareCnt);
    PipeBarrier<PIPE_V>();
    // 量化参数为0的位置不做量化
    Select(tempScale, tempMskBuf, tempScale, xTypeMax, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, curScaleCnt);
    Duplicate(tempWorkTiles, xTypeMax, curScaleCnt);
    PipeBarrier<PIPE_V>();
    Div(curScale, tempWorkTiles, tempScale, curScaleCnt);
    PipeBarrier<PIPE_V>();
    CompareScalar(tempMskBuf, curScale, 0.0f, AscendC::CMPMODE::NE, compareCnt);
    PipeBarrier<PIPE_V>();
    Select(curScale, tempMskBuf, curScale, 1.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, curScaleCnt);
    PipeBarrier<PIPE_V>();
    // Broadcast到{curScaleCnt, 128}，随后Div一次算出所有量化后的数据
    Broadcast<float, BROADCAST_DIM, 1, false>(tempScale, curScale, broadCastDst, broadCastSrc);
    PipeBarrier<PIPE_V>();
    Mul(tilesLocal, tilesLocal, tempScale, padCalCnt);
    PipeBarrier<PIPE_V>();
    Cast(curOutTiles, tilesLocal, RoundMode::CAST_RINT, padCalCnt);
    PipeBarrier<PIPE_V>();
}

template <class T, class U>
__aicore__ inline void DynamicDequant(uint32_t curScaleCnt, uint32_t padCalCnt, LocalTensor<U> outLocal,
                                      LocalTensor<T> tilesLocal, LocalTensor<float> scalesLocal,
                                      LocalTensor<float> tempOut, LocalTensor<float> tempScale)
{
    const uint32_t broadCastDst[BROADCAST_DIM] = {curScaleCnt, static_cast<uint32_t>(TILELEN)};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curScaleCnt, 1};
    Cast(tempOut, tilesLocal, RoundMode::CAST_NONE, padCalCnt);
    Broadcast<float, BROADCAST_DIM, 1, false>(tempScale, scalesLocal, broadCastDst, broadCastSrc);
    PipeBarrier<PIPE_V>();
    if (!std::is_same<U, float>::value) {
        Div(tempOut, tempOut, tempScale, padCalCnt);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, tempOut, RoundMode::CAST_RINT, padCalCnt);
    } else {
        Div(outLocal, tempOut.template ReinterpretCast<U>(), tempScale.template ReinterpretCast<U>(), padCalCnt);
    }
    PipeBarrier<PIPE_V>();
}

template <class XType, class YType>
__aicore__ inline uint32_t GetMaxProcRows(bool isQuant, uint32_t calBuffSize)
{
    uint32_t curUbSize = isQuant ? TOTAL_UB_SIZE - calBuffSize - COMPARE_ALIGN_LEN + 1 - UB_DATABLOCK + 1 :
                                   TOTAL_UB_SIZE - calBuffSize - UB_DATABLOCK + 1;
    // 将ub空间按照各个buf所占比例切分
    uint32_t ubDenom =
        isQuant ? 3 * TILELEN * sizeof(float) + TILELEN * sizeof(XType) + sizeof(float) + sizeof(uint8_t) :
                  TILELEN * sizeof(XType) + sizeof(float) + TILELEN * sizeof(YType) + 2 * TILELEN * sizeof(float);
    ubDenom *= DOUBLE_BUFFER;
    return curUbSize / ubDenom;
}
} // MATMUL_ALL_REDUCE_DYNAMIC_QUANT_UTILS_H

#endif
