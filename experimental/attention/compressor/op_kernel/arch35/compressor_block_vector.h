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

#ifndef COMPRESSOR_BLOCK_VECTOR_H
#define COMPRESSOR_BLOCK_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../compressor_comm.h"
#include "vf/vf_softmax.h"

using namespace AscendC;

namespace Compressor 
{
constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k

template<typename COMP> class CompressorBlockVector {
public:
    using T = float;

    __aicore__ inline CompressorBlockVector(){};
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
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *kvStateOut,
        __gm__ uint8_t *scoreStateOut);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void ComputeVec1(const RunInfo &info, LocalTensor<T> &mmResLeft, LocalTensor<T> &mmResRight);
    __aicore__ inline void ComputeVec2(RunInfo& info);
    __aicore__ inline void SplitCoreV2(RunInfo& info);
    __aicore__ inline void CopyFinalResultOut(RunInfo& info);


protected:
    GlobalTensor<T> vec1ResGm_;
    TBuf<TPosition::VECCALC> kvBuff_;
    TBuf<TPosition::VECCALC> scoreBuff_;

private:
    __aicore__ inline uint32_t GetStartPos(uint32_t index);
    __aicore__ inline uint32_t GetSeqLength(uint32_t index);
    __aicore__ inline uint32_t GetBsLength(uint32_t index);
    __aicore__ inline void WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx,
    uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ReadFromCacheState(LocalTensor<T> &output, GlobalTensor<T> &state, uint32_t batchIdx,
    uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd);
    __aicore__ inline void ProcessSingleBatch(uint32_t batchIdx, uint64_t batchStartSeqIdx, uint64_t batchEndSeqIdx,
    uint32_t dLoop, uint32_t dealDSize, LocalTensor<T> &mmResRight, LocalTensor<T> &mmResLeft);
    

    static constexpr uint32_t v2MBaseSize = 16; // Tc块数量：32 * 1024 / (512 * 4)
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t preStartPosIdx_ = 0;
    uint32_t accSeqLength_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t preActSeqIdx_ = 0;
    uint32_t mmResColSize_ = 128;
    uint32_t v2TcStart_ = 0;
    uint32_t v2TcEnd_ = 0;
    uint32_t mmResBaseOffset_ = 0;
    ConstInfo constInfo_ = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<T> kvStateGm_;
    GlobalTensor<T> scoreStateGm_;
    GlobalTensor<T> apeGm_;


    
};

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
}

template <typename COMP> __aicore__ inline void CompressorBlockVector<COMP>::Init(
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
        __gm__ uint8_t *scoreStateOut)
{
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    kvStateGm_.SetGlobalBuffer((__gm__ T *)kvStateOut);
    scoreStateGm_.SetGlobalBuffer((__gm__ T *)scoreStateOut);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
}

template <typename COMP> __aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(kvBuff_, BLOCK_VEC_BASE_BUFFER_SIZE);
    pipe->InitBuffer(scoreBuff_, BLOCK_VEC_BASE_BUFFER_SIZE);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetStartPos(uint32_t index)
{
    if (preStartPosIdx_ != index) {
        curStartPos_ = startPosGm_.GetValue(index);
        preStartPosIdx_ = index;
        return curStartPos_;
    } else {
        return curStartPos_;
    }
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
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetBsLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(index);
    } else {
        return index * constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::WriteToCacheState(GlobalTensor<T> &state, LocalTensor<T> &input, uint32_t batchIdx, uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStart, uint32_t dEnd)
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
            // 拷贝参数还需要调整
            DataCopyParams copyParams {static_cast<uint16_t>(copyRowCnt), static_cast<uint16_t>(dEnd - dStart), constInfo_.dBaseSize, static_cast<uint16_t>(2 * (dEnd - dStart))};
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
            // print error log
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * (dEnd - dStart) + remainRowCnt * (dEnd - dStart) + dStart;
        // 拷贝参数还需要调整
        DataCopyParams copyParams {static_cast<uint16_t>(copyRowCnt), static_cast<uint16_t>(dEnd - dStart),
            static_cast<uint16_t>(2 * (dEnd - dStart)), static_cast<uint16_t>(2 * (dEnd - dStart))};
        DataCopy(output, state[stateOffset], copyRowCnt);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP> __aicore__ inline void CompressorBlockVector<COMP>::ProcessSingleBatch(uint32_t batchIdx, uint64_t batchStartSeqIdx, uint64_t batchEndSeqIdx,
    uint32_t dLoop, uint32_t dealDSize, LocalTensor<T> &mmResRight, LocalTensor<T> &mmResLeft)
{
    uint32_t seqIdx = batchStartSeqIdx;
    uint64_t seqUsed = GetSeqLength(batchIdx); // seqused有效的话还需要调整
    uint64_t startPos = GetStartPos(batchIdx);
    uint64_t startSeqIdx = 0;
    uint64_t endSeqIdx = 0;
    uint32_t dStartIdx = 0;
    uint32_t dEndIdx = 0;
    uint64_t baseOffset = GetBsLength(batchIdx);
    uint64_t compressSeqId = (startPos + seqUsed) / constInfo_.cmpRatio * constInfo_.cmpRatio;
    uint32_t copyUbBaseOffset = 0;
    LocalTensor<T> kvLocal = kvBuff_.Get<T>();
    LocalTensor<T> scoreLocal = scoreBuff_.Get<T>();
    while (seqIdx < batchEndSeqIdx) {
        // 首次处理半个r，中间处理完整r，结尾半个r
        startSeqIdx = startPos + seqIdx;
        endSeqIdx = startSeqIdx / constInfo_.cmpRatio * constInfo_.cmpRatio + constInfo_.cmpRatio;
        // seqIdx和batchStartSeqIdx为局部索引
        uint64_t startOffset = mmResBaseOffset_ + (seqIdx - batchStartSeqIdx) * constInfo_.dBaseSize; //(baseOffset + (startSeqIdx - startPos)) * constInfo_.dBaseSize;
        // uint64_t endOffset = baseOffset + (endSeqIdx - startPos);
        uint64_t endOffset = baseOffset + (endSeqIdx - startPos);
        bool isSaveState = false;
        bool isCompress = false;
        bool isCopy = true;
        if (endSeqIdx > (startPos + seqUsed) && (batchEndSeqIdx == seqUsed)) {
            // 1.说明当前batch尾部在处理的基本块中，且结尾非完整r块，本次不需要压缩，不需要拷贝到UB,其他场景都需要拷贝
            endSeqIdx = startPos + seqUsed;
            isCopy = false;
        }
        if (startPos == startSeqIdx && (endSeqIdx - startSeqIdx) < constInfo_.cmpRatio) {
            // batch的头小于r,用来和上个的尾结合为一个r，需要将上面补充的数据拷贝过来
            DataCopyParams copyParams {(startPos % constInfo_.cmpRatio), static_cast<uint16_t>(dealDSize), mmResColSize_, static_cast<uint16_t>(2 * dealDSize)};
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将kv的右侧从mm中拷贝过来，源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将score的右侧从mm中拷贝过来,源数据d从64或者96
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResLeft[startOffset + dealDSize * dLoop], copyParams); // 将kv的左侧从mm中拷贝过来,源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResLeft[startOffset + dealDSize * dLoop ], copyParams); // 将score的左侧从mm中拷贝过来,源数据d从64或者96
            copyUbBaseOffset += (startPos % constInfo_.cmpRatio) * dealDSize;        }
        if (isCopy) {
            // 除尾块之外的用于本次压缩的存到UB中, 以右侧索引为准，左侧拷贝同样的seqidx
            DataCopyParams copyParams {static_cast<uint16_t>(endSeqIdx - startSeqIdx), static_cast<uint16_t>(dealDSize), mmResColSize_, static_cast<uint16_t>(2 * dealDSize)};
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将kv的右侧从mm中拷贝过来，源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset], mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], copyParams); // 将score的右侧从mm中拷贝过来,源数据d从64或者96
            DataCopy(kvLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResLeft[startOffset + dealDSize * dLoop], copyParams); // 将kv的左侧从mm中拷贝过来,源数据d从0或者32
            DataCopy(scoreLocal[mmResBaseOffset_ + copyUbBaseOffset + dealDSize], mmResLeft[startOffset + dealDSize * dLoop], copyParams); // 将score的左侧从mm中拷贝过来,源数据d从64或者96
            copyUbBaseOffset += (endSeqIdx - startSeqIdx) * dealDSize; 
        }
        dStartIdx = dealDSize * dLoop;
        dEndIdx = dStartIdx + dealDSize;
        if (startSeqIdx >= (compressSeqId - constInfo_.cmpRatio) && endSeqIdx <= (batchEndSeqIdx + startPos) && (batchEndSeqIdx <= seqUsed)) {
            // 第1种场景：尾块在当前处理的基本块中，最后一个尾块和上一个r块需要存入state, 
            isSaveState = true;
        }
        if ((batchEndSeqIdx + startPos) == endSeqIdx && startSeqIdx < (batchEndSeqIdx + startPos) && (seqUsed - batchEndSeqIdx) <  constInfo_.cmpRatio) {
            // 第2种场景：最后一个完整的r在本次处理的基本块中
            isSaveState = true;
        }
        if (batchEndSeqIdx - batchStartSeqIdx == constInfo_.cmpRatio && ((seqUsed - batchEndSeqIdx < constInfo_.cmpRatio) || (seqUsed == batchEndSeqIdx))) {
            // 第3种场景：本次处理的首尾只有一个r
            isSaveState = true;
            endSeqIdx = (batchEndSeqIdx + startPos);
        }
        // todo:第四种场景，最后一个batch的尾块在第一个batch的首位的左边
        if (startSeqIdx < compressSeqId) {
            isCompress = true;
        }
        if (isSaveState) {
            // 左和右都需要考虑
            // 1.存右边
            WriteToCacheState(kvStateGm_, mmResRight[startOffset + dealDSize * dLoop], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            WriteToCacheState(scoreStateGm_, mmResRight[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            // 2.存左边
            WriteToCacheState(kvStateGm_, mmResLeft[startOffset + dealDSize * dLoop - constInfo_.cmpRatio * constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            WriteToCacheState(scoreStateGm_, mmResLeft[startOffset + dealDSize * dLoop + constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
            // 2. 右边存最后一个尾块
            if ((batchStartSeqIdx + startPos) == startPos) {
                // 第4种情况，左上角有个r块需要存
                WriteToCacheState(kvStateGm_, mmResLeft[startOffset], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);
                WriteToCacheState(scoreStateGm_, mmResLeft[startOffset + constInfo_.dBaseSize], batchIdx, startSeqIdx, endSeqIdx, dStartIdx, dEndIdx);  
            }
        }
        if (isCompress) {
            uint32_t copyStartSeqId = 0;
            uint32_t copyEndSeqId = 0;
            uint32_t coffId = coff_ - 1;
            uint32_t dStart = coffId * constInfo_.dBaseSize + dLoop * dealDSize;
            uint32_t dEnd = (coffId + 1) * constInfo_.dBaseSize + dLoop * dealDSize;
            uint32_t cntFromState = 0;
            // 拷贝右边数据
            if (startPos == startSeqIdx) {
                // batch从起始位置开始
                cntFromState = startPos % constInfo_.cmpRatio;
                if (cntFromState > 0) {
                    copyStartSeqId = startPos - cntFromState;
                    copyEndSeqId = startPos;
                    // 第一块拷贝
                    ReadFromCacheState(kvLocal[mmResBaseOffset_ + dealDSize], kvStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd); // 拷贝的右边
                    ReadFromCacheState(scoreLocal[mmResBaseOffset_ + dealDSize], scoreStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd); // 拷贝的右边
                }
            }
            if (coff_ == 2) {  // 拷贝左边数据
                coffId = 0;
                dStart = coffId * constInfo_.dBaseSize  + dLoop * dealDSize ;
                dEnd = (coffId + 1) * constInfo_.dBaseSize  + dLoop * dealDSize;
                cntFromState = 0;
                if (startPos == startSeqIdx) {
                    // 拷贝左边第一块完整块
                    cntFromState = constInfo_.cmpRatio;
                    if (startPos >= constInfo_.cmpRatio) {
                        copyStartSeqId = startPos - startPos % constInfo_.cmpRatio;
                        copyEndSeqId = copyStartSeqId + cntFromState;
                        ReadFromCacheState(kvLocal[mmResBaseOffset_], kvStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                        ReadFromCacheState(scoreLocal[mmResBaseOffset_], scoreStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                    }
                } else if (startSeqIdx - constInfo_.cmpRatio < startPos) {
                        // 左边数据需要拷贝一个尾块和一个完整r块
                        cntFromState = startPos % constInfo_.cmpRatio;
                        if (cntFromState > 0) {
                            copyStartSeqId = startPos - startPos % constInfo_.cmpRatio;
                            copyEndSeqId = startPos + constInfo_.cmpRatio;
                            ReadFromCacheState(kvLocal[(copyEndSeqId - copyStartSeqId) * dealDSize], kvStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                            ReadFromCacheState(scoreLocal[(copyEndSeqId - copyStartSeqId) * dealDSize], scoreStateGm_, batchIdx, copyStartSeqId, copyEndSeqId, dStart, dEnd);
                        }
                    }
            }
        }
        seqIdx = endSeqIdx - startSeqIdx;
    }
}

template <typename COMP> __aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const RunInfo &info, LocalTensor<T> &mmResLeft, LocalTensor<T> &mmResRight)
{
    uint32_t scLoopTimes = 0;
    uint32_t dLoopTimes = 0;
    uint32_t splitSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.cmpRatio * coff_ * sizeof(T));
    uint32_t dealDSize = constInfo_.dBaseSize;
    if (splitSize < constInfo_.headDim) {
        scLoopTimes = 1;
        dLoopTimes = (constInfo_.headDim + (splitSize - 1)) / splitSize;
        dealDSize = constInfo_.dBaseSize / dLoopTimes;
    } else {
        dLoopTimes = 1;
        scLoopTimes = splitSize / constInfo_.dBaseSize;
    }
    for (uint32_t i = 0; i < scLoopTimes; i++) {
        for (uint32_t j = 0; j < dLoopTimes; j++) {
            for (uint32_t k = info.bStart; k < info.bEnd; k++) {
                // 从UB拷贝到32k空间
                // 存state
                // 从state取
                // overlap
                ProcessSingleBatch(k, info.bStartSeqIdx, info.bEndSeqIdx, j, dealDSize, mmResRight, mmResLeft);
                mmResBaseOffset_ += (info.bEndSeqIdx - info.bStartSeqIdx) * constInfo_.dBaseSize;
            }
            // AddVF<T>()
            //AddVFImpl<T>(outputAddr, inputAddr, aptAddr, regSplitNum, regLeftNum, loopCnt, loopLeft, loadAlignParam0, loadAlignParam1);
            // add -> softmax->reducesum
        }
    }

}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(RunInfo& info)
{
    // vec2核内基本块大小：D不切，S方向切多大？
    SplitCoreV2(info);
    for (uint32_t v2MProcessPos = v2TcStart_, dealSize = v2MBaseSize; v2MProcessPos < v2TcEnd_; v2MProcessPos += v2MBaseSize) {
        if (v2MProcessPos == v2TcEnd_ - 1) {
            dealSize = v2TcEnd_ - v2MProcessPos;
        }
        // RmsNorm
        // rope
        // CopyOut
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::SplitCoreV2(RunInfo& info)
{
    // BS分64vec核；累积N个基本块数据后做vec2，N=2；
    // syncAll前各个vec核写入workspace是按照实际大小（Tc = N * m/r,D=512）
    // 每次进行v2计算都会根据当前情况将workspace中的数据重新分核
    // Input: syncAll前各个vec核写入workspace是按照实际大小(m方向)
    // Output: 每个vec核的起止位置

    uint32_t coreNum = GetBlockNum();
    uint32_t currCoreIdx = GetBlockIdx();
    uint32_t usedCoreNum = 0;
    // 1.计算总vec2基本块数量
    uint64_t totalBaseNum = (constInfo_.coreGroupNum * constInfo_.nSize * constInfo_.tcBaseSize) / v2MBaseSize; // TODO:不是按照实际数据量计算，暂时按照m方向完整基本块计算数据量 
    // 2.每个vec核上分到的数据量
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    }else {
        usedCoreNum = totalBaseNum;
    }
    if(currCoreIdx >= usedCoreNum) {
        return;
    }
    // 3.计算每个vec核的起始结束位置
    v2TcStart_ = currCoreIdx == 0 ? 0 : currCoreIdx * avgBaseNum * v2MBaseSize;
    v2TcEnd_ = v2TcStart_ + avgBaseNum * v2MBaseSize;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CopyFinalResultOut(RunInfo& info)
{

}



} // namespace Compressor

#endif // COMPRESSOR_BLOCK_VECTOR_H