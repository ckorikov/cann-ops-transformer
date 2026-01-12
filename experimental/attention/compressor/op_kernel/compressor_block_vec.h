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
 * \file fia_block_vec_nonquant.h
 * \brief
 */
#ifndef COMPRESSOR_BLOCK_VEC_H
#define FIA_BLOCK_VEC_NONQUANT_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../memory_copy.h"

using namespace AttentionCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP> class CompressorBlockVector{
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k

    __aicore__ inline CompressorBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
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
        __gm__ uint8_t *hadamard,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1();
    __aicore__ inline uint32_t GetSeqLength(uint32_t index);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ReadFromCacheState(LocalTensor<T> &output, GlobalTensor<T> &state, uint32_t batchIdx, uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ProcessSingleBatch(uint32_t batchIdx);

protected:
    GlobalTensor<T> vec1ResGm_;
    TBuf<TPosition::VECCALC> shareBuffer_;

private:
    __aicore__ inline uint32_t GetStartPos(uint32_t index);
    __aicore__ inline uint32_t GetSeqLength(uint32_t index);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t preStartPosIdx_ = 0;
    uint32_t accSeqLength_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t preActSeqIdx_ = 0;
    ConstInfo constInfo_ = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> blockTableGm_;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::Init(
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
        __gm__ uint8_t *hadamard,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut)
{
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(shareBuffer_, BLOCK_VEC_BASE_BUFFER_SIZE);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        if (preActSeqIdx_ != index) {
            preActSeqIdx_ = index;
            if (index == 0) {
                accSeqLength_ = cuSeqlensGm_.GetValue(index + 1);
                return accSeqLength_;
            } else {
                uint32_t tmpSeqLength = accSeqLength_;
                accSeqLength_ = cuSeqlensGm_.GetValue(index + 1);
                return accSeqLength_ - tmpSeqLength;
            }
        } else {
            return curActSeqLength_;
        }
    } else {
        return constInfo.sSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dEnd)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = 0;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        curSeqIdx = startSeqIdx + copyFinishRowCnt;
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm_.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * (dEnd - dStart) + remainRowCnt * (dEnd - dStart) + dStart;
        if (idInBlockTable == 0) {
            DataCopy(state[stateOffset], input, copyRowCnt);
        }
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ReadFromCacheState(LocalTensor<T> &output, GlobalTensor<T> &state, uint32_t batchIdx, uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = 0;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        curSeqIdx = startSeqIdx + copyFinishRowCnt;
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm_.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }
        if (idInBlockTable == 0) {
            // error log
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * (dEnd - dStart) + remainRowCnt * (dEnd - dStart) + dStart;
        DataCopy(output, state[stateOffset], copyRowCnt);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ProcessSingleBatch(uint32_t batchIdx)
{
    uint32_t seqIdx = 0;
    uint64_t seqUsed = GetSeqLength(batchIdx);
    uint64_t startPos = GetStartPos(batchIdx);
    uint64_t startSeqIdx = 0;
    uint64_t endSeqIdx = 0;
    while (seqIdx < seqUsed) {
        startSeqIdx = startPos + seqIdx;
        endSeqIdx = startSeqIdx / constInfo_.cmpRatio * constInfo_.cmpRatio + constInfo_.cmpRatio;
        if (endSeqIdx > (startPos + seqUsed)) {
            endSeqIdx = startPos + seqUsed;
        }
        uint64_t baseOffset = batchIdx * constInfo_.sSize;
        uint64_t startOffset = baseOffset + (startSeqIdx - startPos);
        uint64_t endOffset = baseOffset + (endSeqIdx - startPos);
    }
}

template <typename COMP>
 __aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const RunInfo &info, LocalTensor<T> &mmResLeft, LocalTensor<T> &mmResRight)
{
    uint32_t scLoopTimes = 0;
    uint32_t dLoopTimes = 0;
    uint32_t splitSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.cmpRatio * coff_ * sizeof(T));
    if (splitSize < constInfo_.headDim) {
        scLoopTimes = 1;
        dLoopTimes = (constInfo_.headDim + (splitSize - 1)) / splitSize;
    } else {
        dLoopTimes = 1;
        scLoopTimes = splitSize / constInfo_.headDim;
    }
    for (uint32_t i = 0; i < scLoopTimes; i++) {
        for (uint32_t j = 0; j < dLoopTimes; j++) {
            for (uint32_t k = info.bStart; k < info.bEnd; k++) {
                // 从UB拷贝到32k空间
                // 存state
                // 从state取
                // overlap
                
            }
        }
    }

}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(const RunInfo &info)
{
    
}

#endif // COMPRESSOR_BLOCK_VECTOR_H