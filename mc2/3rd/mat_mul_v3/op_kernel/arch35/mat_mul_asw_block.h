/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file mat_mul_asw_block.h
 * \brief
 */
#ifndef MMV3_MATMUL_ASW_BLOCK_H
#define MMV3_MATMUL_ASW_BLOCK_H

#include "../mat_mul_v3_common.h"
#include "mat_mul_tiling_data.h"

namespace MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

constexpr uint64_t FP32_SPLIT_K_THRESHOLD = 8192UL;

struct AswBlockOffset {
    uint64_t offsetA = 0UL;
    uint64_t offsetB = 0UL;
    uint64_t offsetC = 0UL;
    uint64_t offsetBias = 0UL;
};

struct AswBlockArgs {
    uint64_t index = 0UL;
    uint64_t mCntIndex = 0UL;
    uint64_t nCntIndex = 0UL;
    uint64_t mCnt = 0UL;
    uint64_t nCnt = 0UL;
    uint64_t totalCnt = 0UL;
    uint64_t blockBaseM = 0UL;
    uint64_t blockBaseN = 0UL;
    uint64_t mBaseTailCnt = 0UL;
    uint64_t nBaseTailCnt = 0UL;
    uint64_t mBaseNormCnt = 0UL;
    uint64_t nBaseNormCnt = 0UL;
    uint64_t mBaseTail = 0UL;
    uint64_t nBaseTail = 0UL;
    uint64_t mBaseTailMain = 0UL;
    uint64_t mBaseTailLast = 0UL;
    uint64_t nBaseTailMain = 0UL;
    uint64_t nBaseTailLast = 0UL;
    uint64_t mBaseSplitCnt = 0UL;
    uint64_t nBaseSplitCnt = 0UL;
    uint64_t totalSplitCnt = 0UL;
    uint64_t mSplitAddrOffset = 0UL;
    uint64_t nSplitAddrOffset = 0UL;
    uint64_t singleCoreM = 0UL;
    uint64_t singleCoreN = 0UL;
    uint64_t round = 0UL;
    uint64_t roundToReverse = 0UL;
    uint64_t mainRow = 0UL;
    uint64_t mainWindow = 0UL;
    uint64_t tailWindow = 0UL;
    uint64_t kbAlignSize = 0UL;
    uint64_t nAlignSize = 0UL;

    uint64_t splitKRound = 1UL;
    uint64_t singleCoreSplitK = 0UL;
    uint64_t singleShapeKTail = 0UL;
};

class MatmulAswBlock {
public:
    __aicore__ inline MatmulAswBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    template <class A_TYPE, class B_TYPE>
    __aicore__ inline void LoadBalanceInit();
    __aicore__ inline uint64_t GetNewBlockIdx(uint64_t roundIdx);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();
    template <class A_TYPE, class B_TYPE>
    __aicore__ inline void CalcSplitKGMOffset(uint64_t splitKIndex);

public:
    AswBlockOffset offset_;
    AswBlockArgs params_;
    const MatMulV3TilingData *matmulTilingData_;

private:
    const uint64_t WINDOW_LEN = 4UL;
};

template <class A_TYPE, class B_TYPE>
__aicore__ inline void MatmulAswBlock::LoadBalanceInit()
{
    params_.mBaseTailMain = params_.mBaseTailCnt == 1UL ? params_.mBaseTail :
        MMV3CeilAlign(params_.mBaseTail / params_.mBaseTailCnt, BLOCK_BYTE_SIZE / sizeof(typename A_TYPE::T));
    params_.mBaseTailLast = params_.mBaseTail - (params_.mBaseTailCnt - 1UL) * params_.mBaseTailMain;
    params_.nBaseTailMain = params_.nBaseTailCnt == 1UL ? params_.nBaseTail :
        MMV3CeilAlign(params_.nBaseTail / params_.nBaseTailCnt, BLOCK_BYTE_SIZE / sizeof(typename B_TYPE::T));
    params_.nBaseTailLast = params_.nBaseTail - (params_.nBaseTailCnt - 1UL) * params_.nBaseTailMain;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulAswBlock::Init(const void *tilingData)
{
    matmulTilingData_ = static_cast<const MatMulV3TilingData *>(tilingData);
    params_.index = 0UL;
    params_.singleCoreM = 0UL;
    params_.singleCoreN = 0UL;
    params_.mSplitAddrOffset = 0UL;
    params_.nSplitAddrOffset = 0UL;
    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->tCubeTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->tCubeTiling.baseN);
    params_.mCnt = (matmulTilingData_->tCubeTiling.M + params_.blockBaseM - 1UL) / params_.blockBaseM; // m方向base块数
    params_.nCnt = (matmulTilingData_->tCubeTiling.N + params_.blockBaseN - 1UL) / params_.blockBaseN; // n方向base块数
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.mBaseTailCnt = static_cast<uint64_t>(matmulTilingData_->mBaseTailCnt);
    params_.nBaseTailCnt = static_cast<uint64_t>(matmulTilingData_->nBaseTailCnt);
    params_.mBaseNormCnt = params_.mCnt - params_.mBaseTailCnt;
    params_.nBaseNormCnt = params_.nCnt - params_.nBaseTailCnt;
    // m方向上的base尾块
    params_.mBaseTail = matmulTilingData_->tCubeTiling.M - params_.mBaseNormCnt * params_.blockBaseM;
    // n方向上的base尾块
    params_.nBaseTail = matmulTilingData_->tCubeTiling.N - params_.nBaseNormCnt * params_.blockBaseN;
    LoadBalanceInit<A_TYPE, B_TYPE>();
    params_.round = (params_.totalCnt + matmulTilingData_->tCubeTiling.usedCoreNum - 1UL) /
                    matmulTilingData_->tCubeTiling.usedCoreNum;
    params_.mainWindow = AscendC::Std::min(WINDOW_LEN, params_.mCnt);         // 主划窗m方向的块个数
    params_.mainRow = params_.mCnt / params_.mainWindow - 1UL;                // 主划窗数量
    params_.tailWindow = params_.mCnt - params_.mainRow * params_.mainWindow; // 尾划窗m方向的块个数

    params_.mBaseSplitCnt = matmulTilingData_->mTailCnt;
    params_.nBaseSplitCnt = matmulTilingData_->nTailCnt;
    params_.totalSplitCnt = params_.mBaseSplitCnt * params_.nBaseSplitCnt;
    using B_T = typename B_TYPE::T;
    params_.kbAlignSize = (B_TYPE::isTrans) ? BLOCK_BYTE_SIZE / sizeof(B_T) : BLOCK_SIZE;
    params_.nAlignSize = (B_TYPE::isTrans) ? BLOCK_SIZE : BLOCK_BYTE_SIZE / sizeof(B_T);

    params_.splitKRound = 1UL;
    params_.singleCoreSplitK = matmulTilingData_->tCubeTiling.singleCoreK;
    params_.singleShapeKTail = matmulTilingData_->tCubeTiling.singleCoreK;
    constexpr bool isFp32 = std::is_same_v<typename C_TYPE::T, float>;
    // 如果是fp32且singleCoreK大于8192且B矩阵不是NZ格式，需要单核切K保精度，否则不需要切K
    if (isFp32 && !matmulTilingData_->isHf32 && B_TYPE::format == CubeFormat::ND &&
        matmulTilingData_->tCubeTiling.singleCoreK > FP32_SPLIT_K_THRESHOLD) {
        params_.splitKRound = MMV3DivCeil(matmulTilingData_->tCubeTiling.singleCoreK, FP32_SPLIT_K_THRESHOLD);
        params_.singleCoreSplitK = MMV3CeilAlign(
            MMV3DivCeil(matmulTilingData_->tCubeTiling.singleCoreK, params_.splitKRound), ALIGN_BYTE / DATA_SIZE_FP32);
        params_.singleShapeKTail = matmulTilingData_->tCubeTiling.singleCoreK % params_.singleCoreSplitK;
        if (params_.singleShapeKTail == 0UL) {
            params_.singleShapeKTail = params_.singleCoreSplitK;
        }
    }
}

// aswt模板当m切块是2或者是4的倍数，则可以偏移分核进行负载均衡
__aicore__ inline uint64_t MatmulAswBlock::GetNewBlockIdx(uint64_t roundIdx)
{
    uint64_t newBlockIdx = GetBlockIdx();
    newBlockIdx = (roundIdx == params_.round - 1UL) ? (newBlockIdx / params_.totalSplitCnt) : newBlockIdx;
    return newBlockIdx;
}

__aicore__ inline void MatmulAswBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    params_.index = newBlockIdx + roundIdx * matmulTilingData_->tCubeTiling.usedCoreNum;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mainWindow;
    if (rowIdx < params_.mainRow) {
        params_.mCntIndex = rowIdx * params_.mainWindow + params_.index % params_.mainWindow;
        params_.nCntIndex = (params_.index / params_.mainWindow) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mainWindow * params_.nCnt;
        params_.mCntIndex = params_.mainRow * params_.mainWindow + tailIndex % params_.tailWindow;
        params_.nCntIndex = (tailIndex / params_.tailWindow) % params_.nCnt;
    }
    // mod 2 means even row, need reverse scan
    if (rowIdx % NUM_TWO != 0UL) {
        params_.nCntIndex = params_.nCnt - 1UL - params_.nCntIndex;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.blockBaseM;
    if (params_.mCntIndex >= params_.mBaseNormCnt) {
        params_.singleCoreM =
            (params_.mCntIndex >= (params_.mCnt - 1UL)) ? params_.mBaseTailLast : params_.mBaseTailMain;
    }

    params_.singleCoreN = params_.blockBaseN;
    if (params_.nCntIndex >= params_.nBaseNormCnt) {
        params_.singleCoreN =
            (params_.nCntIndex >= (params_.nCnt - 1UL)) ? params_.nBaseTailLast : params_.nBaseTailMain;
    }

    if (roundIdx == params_.round - 1UL && (params_.mBaseSplitCnt != 1UL || params_.nBaseSplitCnt != 1UL)) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + params_.mBaseSplitCnt - 1UL) / params_.mBaseSplitCnt;
        uint64_t singleCoreNSplit = (params_.singleCoreN + params_.nBaseSplitCnt - 1UL) / params_.nBaseSplitCnt;
        if constexpr (B_TYPE::format != CubeFormat::ND) {
            singleCoreNSplit = MMV3CeilAlign(singleCoreNSplit, params_.nAlignSize);
        }
        params_.mBaseSplitCnt = MMV3DivCeil(params_.singleCoreM, singleCoreMSplit);
        params_.nBaseSplitCnt = MMV3DivCeil(params_.singleCoreN, singleCoreNSplit);
        uint64_t curBlockIdx = GetCurrentBlockIdx();
        uint64_t mSplitIdx = (curBlockIdx % params_.totalSplitCnt) % params_.mBaseSplitCnt;
        uint64_t nSplitIdx = (curBlockIdx % params_.totalSplitCnt) / params_.mBaseSplitCnt;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0UL;
            params_.singleCoreN = 0UL;
            return;
        }
        if (mSplitIdx + 1UL == params_.mBaseSplitCnt) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (nSplitIdx + 1UL == params_.nBaseSplitCnt) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulAswBlock::CalcGMOffset()
{
    uint64_t mOffset = params_.mCntIndex * params_.blockBaseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset;
    if (params_.mCntIndex > params_.mBaseNormCnt) {
        mOffset = mOffset - (params_.mCntIndex - params_.mBaseNormCnt) * (params_.blockBaseM - params_.mBaseTailMain);
    }
    if (params_.nCntIndex > params_.nBaseNormCnt) {
        nOffset = nOffset - (params_.nCntIndex - params_.nBaseNormCnt) * (params_.blockBaseN - params_.nBaseTailMain);
    }

    if constexpr (A_TYPE::isTrans) {
        offset_.offsetA = mOffset;
    } else {
        offset_.offsetA = mOffset * matmulTilingData_->tCubeTiling.Ka;
    }

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if constexpr (B_TYPE::isTrans) {
            offset_.offsetB = nOffset * matmulTilingData_->tCubeTiling.Kb;
        } else {
            offset_.offsetB = nOffset;
        }
    } else {
        if constexpr (B_TYPE::isTrans) {
            offset_.offsetB = nOffset * params_.kbAlignSize;
        } else {
            offset_.offsetB = nOffset * MMV3CeilAlign(matmulTilingData_->tCubeTiling.Kb, params_.kbAlignSize);
        }
    }
    offset_.offsetC = nOffset + mOffset * matmulTilingData_->tCubeTiling.N;
    if (matmulTilingData_->tCubeTiling.isBias) {
        offset_.offsetBias = nOffset;
    }
}

template <class A_TYPE, class B_TYPE>
__aicore__ inline void MatmulAswBlock::CalcSplitKGMOffset(uint64_t splitKIndex)
{
    if (params_.splitKRound == 1) {
        return;
    }
    uint64_t mOffset = params_.mCntIndex * params_.blockBaseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nCntIndex * params_.blockBaseN + params_.nSplitAddrOffset;
    if (params_.mCntIndex > params_.mBaseNormCnt) {
        mOffset = mOffset - (params_.mCntIndex - params_.mBaseNormCnt) * (params_.blockBaseM - params_.mBaseTailMain);
    }
    if (params_.nCntIndex > params_.nBaseNormCnt) {
        nOffset = nOffset - (params_.nCntIndex - params_.nBaseNormCnt) * (params_.blockBaseN - params_.nBaseTailMain);
    }
    if constexpr (A_TYPE::isTrans) {
        offset_.offsetA = mOffset + splitKIndex * params_.singleCoreSplitK * matmulTilingData_->tCubeTiling.M;
    } else {
        offset_.offsetA = mOffset * matmulTilingData_->tCubeTiling.Ka + splitKIndex * params_.singleCoreSplitK;
    }
    if constexpr (B_TYPE::isTrans) {
        offset_.offsetB = nOffset * matmulTilingData_->tCubeTiling.Kb + splitKIndex * params_.singleCoreSplitK;
    } else {
        offset_.offsetB = nOffset + splitKIndex * params_.singleCoreSplitK * matmulTilingData_->tCubeTiling.N;
    }
}

} // namespace MatmulV3Advanced

#endif // MMV3_MATMUL_ASW_BLOCK_H