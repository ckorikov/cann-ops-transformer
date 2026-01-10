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
 * \file compressor_kernel.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_H
#define COMPRESSOR_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"
#include "compressor_comm.h"

using namespace AscendC;

namespace Compressor {

template <typename COMP>
class CompressorKernel {
public:
    __aicore__ inline CompressorKernel(TPipe* pipe, const optiling::CompressorTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut,
        __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t CalcTcSize();
    __aicore__ inline uint32_t GetSeqLength(uint32_t index);
    __aicore__ inline uint32_t GetStartPos(uint32_t index);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline void InitTilingData();
    __aicore__ inline bool IsNeedExcute();
    __aicore__ inline uint32_t GetStartIdx();
    __aicore__ inline uint32_t GetEndIdx();
    __aicore__ inline void ComputeMm1(uint32_t startIdx, uint32_t endIdx);
    __aicore__ inline void ComputeVec1(uint32_t startIdx, uint32_t endIdx);
    __aicore__ inline void ComputeVec2(uint32_t startIdx, uint32_t endIdx);

    TPipe* pipe_;
    const optiling::CompressorTilingData* __restrict tilingData_;
    uint32_t batchSize = 0;
    uint32_t hSize = 0;
    uint32_t sSize = 0;
    uint32_t headDim = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t cmpRatio = 0;
    uint32_t normEps = 0;

    uint32_t blockNum = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    uint32_t mmKVLeftResSize = 0;
    uint32_t mmKVRightResSize = 0;
    uint32_t mmScoreLeftResSize = 0;
    uint32_t mmScoreRightResSize = 0;
    uint32_t vecResSize = 0;

    // 分核信息
    uint32_t coreNum = 24;
    uint32_t dBaseSize = 64;
    uint32_t mBaseSize = 256;
    uint32_t tcSize = 0;
    uint32_t tcBaseSize = 0;
    uint32_t tcBasicBlockNum = 0;
    uint32_t dBasicBlockNum = 0;
    uint32_t coreGroupNum = 0;
    uint32_t singleCoreDealTcBasicNum = 0;

    uint32_t accSeqLength = 0;
    uint32_t curActSeqLength = 0;
    uint32_t curStartPos = 0;
    uint32_t preActSeqIdx = 0;
    uint32_t preStartPosIdx = 0;
    uint32_t bIdx = 0;
    uint32_t sStartIdx = 0;
    uint32_t sEndIdx = 0;
    uint32_t bStart = 0;
    uint32_t sStart = 0;
    uint32_t bEnd = 0;
    uint32_t sEnd = 0;
    uint32_t dEnd = 0;

    uint32_t aiCoreIdx = 0;

    // 常量
    static constexpr uint32_t N = 2;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 6;
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    // TODO fp16
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<int32_t> kvStateGm_;
    GlobalTensor<int32_t> scoreStateGm_;
    GlobalTensor<int32_t> apeGm_;
    GlobalTensor<X_T> ropeSinGm_;
    GlobalTensor<X_T> ropeCosGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
};

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut,
        __gm__ uint8_t *workspace) {
    // printf("[VERSION] 20260109-001\n");
    // printf("CompressorKernel::Init!!!!!\n");

    // TODO CV非1:2需处理 
    if ASCEND_IS_AIV {
        aiCoreIdx = GetBlockIdx() / 2;
    } else {
        aiCoreIdx = GetBlockIdx();
    }

    // GM Init
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    kvStateGm_.SetGlobalBuffer((__gm__ int32_t *)kvState);
    scoreStateGm_.SetGlobalBuffer((__gm__ int32_t *)scoreState);
    apeGm_.SetGlobalBuffer((__gm__ int32_t *)ape);
    ropeSinGm_.SetGlobalBuffer((__gm__ X_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ X_T *)ropeCos);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);

    InitTilingData();

    // 初始化 curActSeqLength、start_pos TODO考虑为None， 
    if (COMP::xLayout == X_LAYOUT::TH) {
        curActSeqLength = cuSeqlensGm_.GetValue(1);
        accSeqLength = curActSeqLength;
        // printf("[Init] curActSeqLength:%u\n", curActSeqLength);
    }

    // 计算分核基本信息
    tcSize = CalcTcSize();
    tcBaseSize = mBaseSize / cmpRatio;
    tcBasicBlockNum = (tcSize + tcBaseSize - 1) / tcBaseSize;       // TC方向的基本块
    dBasicBlockNum = headDim / dBaseSize;                           // D方向的基本块
    coreGroupNum = coreNum / dBasicBlockNum;                        // 核分为多少组
    singleCoreDealTcBasicNum = (tcBasicBlockNum + coreGroupNum - 1) / coreGroupNum; // 处理的最大基本块数量
    // printf("[BASEINFO] tcSize:%u tcBaseSize:%u tcBasicBlockNum:%u dBasicBlockNum:%u coreGroupNum:%u singleCoreDealTcBasicNum:%u\n", tcSize, tcBaseSize, tcBasicBlockNum, dBasicBlockNum, coreGroupNum, singleCoreDealTcBasicNum);

    curStartPos = startPosGm_.GetValue(0);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::InitTilingData() {
    cmpRatio = tilingData_->baseParams.cmpRatio;
    batchSize = tilingData_->baseParams.batchSize;
    mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    dBaseSize = tilingData_->innerSplitParams.dBaseSize;
    headDim = tilingData_->baseParams.headDim;
    hSize = tilingData_->baseParams.hiddenSize;
    ropeHeadDim = tilingData_->baseParams.ropeHeadDim;
    
    blockNum = tilingData_->pageAttentionParams.blockNum;
    blockSize = tilingData_->pageAttentionParams.blockSize;
    maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    mmKVLeftResSize = tilingData_->workspaceParams.mmKVLeftResSize;
    mmKVRightResSize = tilingData_->workspaceParams.mmKVRightResSize;
    mmScoreLeftResSize = tilingData_->workspaceParams.mmScoreLeftResSize;
    mmScoreRightResSize = tilingData_->workspaceParams.mmScoreRightResSize;
    vecResSize = tilingData_->workspaceParams.vecResSize;
    // printf("[TILINGDATA] cmpRatio:%u batchSize:%u mBaseSize:%u dBaseSize:%u\n", cmpRatio, batchSize, mBaseSize, dBaseSize);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::CalcTcSize() {
    uint32_t totalBasicNum = 0;

    for (uint32_t i = 0; i < batchSize; ++i) {
        curStartPos = GetStartPos(i);
        curActSeqLength = GetSeqLength(i);
        totalBasicNum += GetBasicNum();

        // printf("[CalcTcSize] curActSeqLength:%u totalBasicNum:%d\n", curActSeqLength, totalBasicNum);
    }
    
    return totalBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetSeqLength(uint32_t index) {
    // printf("[GetSeqLength] preActSeqIdx:%u index:%u\n", preActSeqIdx, index);
    if (COMP::xLayout == X_LAYOUT::TH) {
        if (preActSeqIdx != index) {
            preActSeqIdx = index;
            if (index == 0) {
                accSeqLength = cuSeqlensGm_.GetValue(index + 1);
                return accSeqLength;
            } else {
                uint32_t tmpSeqLength = accSeqLength;
                accSeqLength = cuSeqlensGm_.GetValue(index + 1);
                return accSeqLength - tmpSeqLength;
            }
        } else {
            return curActSeqLength;
        }
    } else {
        return sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetStartPos(uint32_t index) {
    if (preStartPosIdx != index) {
        curStartPos = startPosGm_.GetValue(index);
        preStartPosIdx = index;
        return curStartPos;
    } else {
        return curStartPos;
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernel<COMP>::IsNeedExcute() {
    if (bIdx == batchSize) {
        return false;
    }
    return true;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetStartIdx() {
    uint32_t totalBasicNum = 0;
    // 在当前batch的seq开始索引位置
    uint32_t startIdx = 0;
    // Tc的开始位置
    uint32_t basicNumStart = (aiCoreIdx / dBasicBlockNum) * tcBaseSize * singleCoreDealTcBasicNum;

    if (bIdx >= batchSize) {
        return startIdx;
    }
    // 第一次遍历计算当前核起始位置
    if (bIdx == 0 && sEndIdx == 0) {
        for (uint32_t i = bEnd; i < batchSize; ++i) {
            if (totalBasicNum == basicNumStart) {
                // printf("[PRINT] bIdx:%u basicNumStart:%u\n", bIdx, basicNumStart);
                bIdx = i;
                sStart = startIdx;
                sStartIdx = startIdx;
                return startIdx;
            }
            curStartPos = GetStartPos(i);
            curActSeqLength = GetSeqLength(i);

            // 加上头块，若有
            uint32_t curBasicNum = 0;
            uint32_t headSize = 0;
            if (curStartPos % cmpRatio != 0) {
                headSize = cmpRatio - curStartPos % cmpRatio;
                curBasicNum++;
            }
            // 加上中间整块及尾块
            curBasicNum += (curActSeqLength - headSize + cmpRatio - 1) / cmpRatio;
            // printf("[PRINT] bIdx:%u basicNumStart:%u headSize:%u basicNumStart:%u curBasicNum:%u  curStartPos:%u, curActSeqLength:%u\n", bIdx, basicNumStart, headSize, basicNumStart, curBasicNum, curStartPos, curActSeqLength);
            if (totalBasicNum + curBasicNum > basicNumStart) {
                bIdx = i;
                bStart = i;
                uint32_t curBasicNumStart = basicNumStart - totalBasicNum;
                if (curBasicNumStart > 0 && headSize > 0) {
                    startIdx = headSize + (curBasicNumStart - 1) * cmpRatio;
                } else {
                    startIdx = curBasicNumStart * cmpRatio;
                }
                sStart = startIdx;
                sStartIdx = startIdx;
                return startIdx;
            }
            totalBasicNum += curBasicNum;
        }
    } else {
        bIdx = bEnd;
        curActSeqLength = GetSeqLength(bIdx);
        if (sEndIdx == curActSeqLength) {
            bIdx ++;
            sStartIdx = startIdx;
            return startIdx;
        } else {
            sStartIdx = sEndIdx;
            return sStartIdx;
        }
    }

    return startIdx;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetBasicNum() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos % cmpRatio != 0) {
        headSize = cmpRatio - curStartPos % cmpRatio;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength - headSize + cmpRatio - 1) / cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetEndIdx() {
    // uint32_t basicNumEnd = aiCoreIdx * tcBaseSize * singleCoreDealTcBasicNum;
    uint32_t accBasicNum = 0;
    uint32_t dealTcNum =  tcBaseSize;
    for (uint32_t i = bIdx; i < batchSize; ++i) {
        bEnd = i;
        if (i == bIdx) {
            curActSeqLength = GetSeqLength(i);
            curStartPos = GetStartPos(i);
            uint32_t curRemainTcNum = 0;
            // 计算起始batch的剩余seq长度 起始位置计算头块
            uint32_t headSize = 0;
            if (curStartPos % cmpRatio != 0) {
                headSize = (cmpRatio - curStartPos % cmpRatio);
            }
            if (sStartIdx == 0) {
                curRemainTcNum = (curActSeqLength - headSize + cmpRatio - 1) / cmpRatio;
                curRemainTcNum = headSize == 0 ? curRemainTcNum : curRemainTcNum + 1;
            } else {
                curRemainTcNum = (curActSeqLength - sStartIdx + cmpRatio - 1) / cmpRatio;
            }
            // printf("[GetEndIdx]  i:%u accBasicNum:%u dealTcNum:%u curRemainTcNum:%u headSize:%u curStartPos:%u curActSeqLength:%u \n", i, accBasicNum, dealTcNum, curRemainTcNum, headSize, curStartPos, curActSeqLength);
            if (curRemainTcNum > dealTcNum) {
                if (sStartIdx == 0) {
                    if (headSize == 0) {
                        sEndIdx = sStartIdx + dealTcNum * cmpRatio;
                    } else {
                        sEndIdx = sStartIdx + headSize + (dealTcNum - 1) * cmpRatio;
                    }
                    
                    return sEndIdx;
                } else {
                    sEndIdx = sStartIdx + dealTcNum * cmpRatio;
                    return sEndIdx;
                }
            } else if (curRemainTcNum == dealTcNum || i == batchSize - 1) {
                sEndIdx = curActSeqLength;
                return sEndIdx;
            } else {
                accBasicNum += curRemainTcNum;
            }
            
        } else {
            curActSeqLength = GetSeqLength(i);
            curStartPos = GetStartPos(i);
            uint32_t curBasicNum = GetBasicNum();
            // printf("[GetEndIdx] accBasicNum:%u curBasicNum:%u dealTcNum:%u\n", accBasicNum, curBasicNum, dealTcNum);
            if (accBasicNum + curBasicNum > dealTcNum) {
                uint32_t headSize = 0;
                if (curStartPos % cmpRatio != 0) {
                    headSize = cmpRatio - curStartPos % cmpRatio;
                }
                uint32_t curBasicNumEnd = dealTcNum - accBasicNum;
                if (headSize == 0) {
                    sEndIdx = curBasicNumEnd * cmpRatio;
                } else {
                    sEndIdx = headSize + (curBasicNumEnd - 1) * cmpRatio;
                }
                sEndIdx = sEndIdx > curActSeqLength ? curActSeqLength : sEndIdx;
                return sEndIdx;
            } else if (accBasicNum + curBasicNum == dealTcNum) {
                sEndIdx = curActSeqLength;
                return sEndIdx;
            }
            accBasicNum += curBasicNum;
        }
        
    }
    return sEndIdx;
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeMm1(uint32_t startIdx, uint32_t endIdx) {
    // printf("[COMPUTE] MM1 bStart:%d bEnd:%d startIdx:%d endIdx:%d\n", bIdx, bEnd, startIdx, endIdx);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeVec1(uint32_t startIdx, uint32_t endIdx) {
    // printf("[COMPUTE] VEC1 bStart:%d bEnd:%d startIdx:%d endIdx:%d\n", bIdx, bEnd, startIdx, endIdx);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeVec2(uint32_t startIdx, uint32_t endIdx) {
    // printf("[COMPUTE] VEC2 bStart:%d bEnd:%d startIdx:%d endIdx:%d\n", bIdx, bEnd, startIdx, endIdx);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Process() {
    // printf("CompressorKernel::Process!!!!!\n");
    
    for (uint32_t i = 0; i < singleCoreDealTcBasicNum; ++i) {
        
        // 获取各切分轴的起始核结束索引
        uint32_t startIdx = GetStartIdx();
        uint32_t endIdx = 0;
        bool isNeedExcute = IsNeedExcute();
        if (isNeedExcute) {
            endIdx = GetEndIdx();
        }
        if ASCEND_IS_AIC {
            if (isNeedExcute) {
                ComputeMm1(startIdx, endIdx);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
            }
        } else {
            if (isNeedExcute) {
                CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
                ComputeVec1(startIdx, endIdx);
            }
            // 累积N个基本块/最后一次循环
            if ((i + 1) % N == 0 || (i + 1) == singleCoreDealTcBasicNum) {
                SyncAll();
                if (isNeedExcute) {
                    ComputeVec2(startIdx, endIdx);
                }
            }
        }
    }

}

} // namespace Compressor

#endif // COMPRESSOR_KERNEL_H