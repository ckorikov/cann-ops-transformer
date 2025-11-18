/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_grad_kl_loss_common.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_COMMON_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

// V0 -> C1 P
constexpr uint8_t SYNC_V0_TO_C1_P_FLAG[2][2] = {{0, 1}, {2, 3}};
// V0 -> C1 SY
constexpr uint8_t SYNC_V0_TO_C1_SY_FLAG[2][2] = {{4, 5}, {6, 7}};
// C1 -> V1 P
constexpr uint8_t SYNC_C1_TO_V1_P_FLAG[2] = {8, 9};
// C1 -> V1 SY
constexpr uint8_t SYNC_C1_TO_V1_SY_FLAG[2] = {10, 11};

constexpr uint8_t SYNC_V1_TO_C2_DW_FLAG[2] = {12, 13};

constexpr uint8_t SYNC_C2_TO_V2_DW_FLAG[2] = {14, 15};
static constexpr uint32_t N_WORKSPACE_SIZE = 1024; // n方向切分

enum class SLILayout
{
    BSND = 0,
    TND = 1
};

enum class SLISparseMode
{
    RightDown = 3
};

struct GatherParams {
    int32_t s2ProcessSize;
    int32_t dValue;
    int32_t dRopeValue;
};

/** @name 模版类型定义
 *  @{
 */
template <typename InputQT, typename InputKT, typename OutT,
	      SLILayout LayoutQT = SLILayout::TND,
          SLILayout LayoutKT = SLILayout::TND,
          SLISparseMode SparseMode = SLISparseMode::RightDown,
          const bool HasRope = false, const bool Deterministic = false,
          typename... Args>
struct SLIType {
    using inputQT = InputQT;
    using inputKT = InputKT;
    using outputT = OutT;
    static constexpr SLILayout inputQLayout = LayoutQT;
    static constexpr SLILayout inputKLayout = LayoutKT;
    static constexpr bool hasRope = HasRope;
    static constexpr bool deterministic = Deterministic;
};
/// @}


/** @name 核内数据结构定义
 *  @{
 */
struct SLIGradKLLossConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_64 = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256 = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512 = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_9K = 9 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_11K = 11 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 64 * 1024;

    /** \brief 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t subBlockIdx;

    /** \brief TilingData中的信息 */
	uint32_t bSize;
	uint32_t n2Size; // 现阶段n2Size默认等于1
    uint32_t gSizeQuery; // 128或者64
    uint32_t gSizeQueryIndex; // 64或者32
    uint32_t s1Size; // 支持泛化
    uint32_t s2Size; // 支持泛化
    uint32_t dSizeQuery; // 默认不带Rope，固定等于512
    uint32_t dSizeQueryIndex; // 默认不带Rope，固定等于128
    uint32_t dSizeQueryRope = 64; // Rope，固定等于64
    uint32_t kSize; // 现阶段只支持2048
    SLISparseMode sparseMode; // 0或者3
    float scaleValue;

    // 轴的乘积提取一些公共信息，减少scalar
    int32_t s2BaseSize;
    int32_t dSizeRope;
    int32_t gatherKeyDbNum;
    int32_t gatherKeyIndexDbNum;
    int32_t gatherKeySize;
    int32_t gatherKeyIndexSize;
    int32_t sparseBlockSize;
};

struct SLIGradKLLossRunInfo {
    uint32_t taskId;
    uint32_t taskIdMod2;
	uint32_t bIdx;
    uint32_t s1Idx;
    uint32_t kIdx;
    int64_t accumS1Idx; // 当前循环累加的T1
    int64_t accumS2Idx; // 当前循环累加的T2
    uint32_t actS1Size; // 当前batch的S1Size
    uint32_t actS2Size; // 当前batch的S2Size
    uint32_t kBaseSize; // k方向切分的基本块大小，在P和KLLoss阶段
    uint32_t kRealSize; // k切分之后的实际大小，在P和KLLoss阶段
    uint32_t kRealSizeAlign8;
    uint32_t kLoopTimes; // k方向的循环次数，在P和KLLoss阶段
    
    uint32_t nBaseSizeP;
    uint32_t nRealSizeP;
    uint32_t nBaseSizeSY;
    uint32_t nRealSizeSY;
    uint32_t nIdxP;
    uint32_t nIdxSY;
    
    // 存放一些offset，减少重复计算
    int32_t s2SparseLen;
    int32_t s2RealSize;
    int32_t s2LoopTimes;
    int32_t s2TailSize;
    int32_t s2Idx;

    // 根据bIdx、s1Idx算出的偏移
    int64_t queryTensorOffset = 0;
    int64_t queryRopeTensorOffset = 0;
    int64_t queryIndexTensorOffset = 0;
};
/// @}


// ================================Util functions==================================
template <typename T> __aicore__ inline T SFAAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T1, typename T2> __aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

template <typename T>
__aicore__ inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_COMMON_H
