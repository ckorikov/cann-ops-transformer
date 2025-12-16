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
 * \file moe_gating_top_k_e_k_fullload.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_E_K_FULLLOAD_H
#define MOE_GATING_TOP_K_E_K_FULLLOAD_H
#include "kernel_operator.h"
#include "common.h"
namespace MoeGatingTopK {
using namespace AscendC;

const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216;
constexpr uint32_t FakerFp32RealFp16Multi = 2;
const int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void GatherV100(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                  const LocalTensor<int32_t>& indexTensor,
                                  const LocalTensor<uint8_t>& sharedTmpBuffer,
                                  const uint32_t count) {
    LocalTensor<int32_t> tempBuffer = sharedTmpBuffer.template ReinterpretCast<int32_t>();
    DataCopy(tempBuffer, indexTensor, count);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t realIndex = static_cast<uint32_t>(tempBuffer.GetValue(index));
        dst.SetValue(index, src.GetValue(realIndex));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

__aicore__ inline void ReduceSumFp32V100(const LocalTensor<float>& dst,
                                         const LocalTensor<float>& src,
                                         const uint32_t count) {
    Duplicate(dst, (float)0, 64);
    RepeatReduceSum<float>(dst, src, 1, count, 0, 1, 1, 8);
}


template <typename T>
class MoeGatingTopKEKFullload {
public:
    __aicore__ inline MoeGatingTopKEKFullload(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearGMForAtomicAdd();
    __aicore__ inline void CopyInBias();
    __aicore__ inline void CopyInX(int64_t progress);
    __aicore__ inline void ComputeX();
    __aicore__ inline void SortInGroup();
    __aicore__ inline void SelectTopKGroupIndex();
    __aicore__ inline void SelectTopKExpertIdx();
    __aicore__ inline void SelectTopKExpertIdxEight();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CopyOut(int64_t progress);

private:
    TPipe *pipe_;

    GlobalTensor<T> biasGm_;
    TBuf<TPosition::VECCALC> biasInBuffer_;

    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECIN, 1> xInQueue_;

    TQue<QuePosition::VECOUT, 1> xSigmoidAddBiasQueue_;
    TQue<QuePosition::VECOUT, 1> xSigmoidQueue_;

    TQue<QuePosition::VECIN, 1> sortedInGroupQueue_;
    TBuf<TPosition::VECCALC> calcTmpBuffer_;
    TBuf<TPosition::VECCALC> sortGroupTmpBuffer_;
    // TBuf<TPosition::VECCALC> sigmoidTmpBuffer_;

    TQue<QuePosition::VECIN, 1> sortedGroupQueue_;

    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TBuf<TPosition::VECCALC> syncTmpSpaceBuffer_;
    LocalTensor<half> indexTensor_;
    // LocalTensor<half> gropedSortedScore_;
    // LocalTensor<half> top2ScoreSumPerGroup_;
    // LocalTensor<half> sortOutGroupBuffer_;
    // LocalTensor<half> reduceSumBuffer_;
    LocalTensor<half> finalSortTemp_;
    LocalTensor<uint8_t> sharedTmpBuffer_;

    GlobalTensor<int32_t> syncTmpSpaceGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> yGmFp32_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> outGm_;

    int64_t blockIdx_;
    int64_t rowCount_;
    int64_t perCoreRowCount_;
    int64_t curCoreRowCount_;
    int64_t expertCount_;
    bool addBias_;
    int64_t k_;
    int64_t kGroup_;
    int64_t groupCount_;
    int64_t groupSelectMode_;
    int64_t renorm_;
    int64_t normType_;
    int64_t outFlag_;
    float routedScalingFactor_;
    float eps_;

    int64_t expertCountAlign_;
    int64_t kAlign_;
    int64_t perGroupExpertCount_;

    uint32_t pRsortBufferToatalSize_;
    uint32_t pRsortBufferPerRepeat_;
    uint32_t pRsoetExBuffer_;
    uint32_t regionProposalByte_ = 16;
    uint32_t syncLen_;

    const MoeGatingTopKTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::ClearGMForAtomicAdd() {

    // support curCoreRowCount_ < 16384 when k_ == 8
    int64_t dataBytes = (curCoreRowCount_ * k_ * sizeof(half) + 31) / 32 * 32;
    LocalTensor<half> tempTensor = sharedTmpBuffer_.template ReinterpretCast<half>();
    Duplicate(tempTensor, (half)0, dataBytes / sizeof(half));
    
    DataCopy(yGm_, tempTensor, {static_cast<uint16_t>(1),
                                static_cast<uint16_t>(dataBytes / 32),
                                static_cast<uint16_t>(0),
                                static_cast<uint16_t>(1)});
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyInBias()
{   LocalTensor<half> biasTensorTemp = sharedTmpBuffer_.template ReinterpretCast<half>();
    LocalTensor<float> biasTensor = biasInBuffer_.Get<float>();
    DataCopy(biasTensorTemp, biasGm_, expertCount_);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    Cast(biasTensor, biasTensorTemp, RoundMode::CAST_NONE, expertCount_);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyInX(int64_t row)
{
    LocalTensor<half> xInLocalTensor = xInQueue_.AllocTensor<half>();
    DataCopy(xInLocalTensor, xGm_[row * expertCount_], expertCount_);
    xInQueue_.EnQue<half>(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::ComputeX()
{
    LocalTensor<half> xInLocalTensor = xInQueue_.DeQue<half>();
    LocalTensor<float> xInLocalTensorFp32 = sharedTmpBuffer_.template ReinterpretCast<float>();
    LocalTensor<float> xBiasTensorFp32 = sharedTmpBuffer_[expertCountAlign_ * 4].template ReinterpretCast<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuffer_[expertCountAlign_ * 4 * 2];
    LocalTensor<float> biasTensor = biasInBuffer_.Get<float>();
    LocalTensor<half> xSigmoidTensor = xSigmoidQueue_.AllocTensor<half>();
    LocalTensor<float> xSigmoidTensorFp32 = xSigmoidTensor[expertCountAlign_].template ReinterpretCast<float>();
    LocalTensor<half> xBiasTensor = xSigmoidAddBiasQueue_.AllocTensor<half>();

    PipeBarrier<PIPE_V>();
    Cast(xInLocalTensorFp32, xInLocalTensor, RoundMode::CAST_NONE, expertCount_);
    Sigmoid(xSigmoidTensorFp32, xInLocalTensorFp32, sharedTmpBuffer, expertCount_);

    if (addBias_) {
        Add(xBiasTensorFp32, xSigmoidTensorFp32, biasTensor, expertCount_);
    } else {
        Adds(xBiasTensorFp32, xSigmoidTensorFp32, static_cast<float>(0), expertCount_);
    }
    Cast(xBiasTensor, xBiasTensorFp32, RoundMode::CAST_NONE, expertCount_);
    xSigmoidAddBiasQueue_.EnQue<half>(xBiasTensor);
    xSigmoidQueue_.EnQue<half>(xSigmoidTensor);
    xInQueue_.FreeTensor(xInLocalTensor);
}
template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SortInGroup()
{
    LocalTensor<half> xBiasTensor = xSigmoidAddBiasQueue_.DeQue<half>();
    LocalTensor<half> sortedInGroupTensor = sortedInGroupQueue_.AllocTensor<half>(); // 组内排序的结果, 后续归并需要
    LocalTensor<half> sortTmpBuffer = sortGroupTmpBuffer_.Get<half>();
    LocalTensor<half> sortTmpBuffer2 = sortTmpBuffer[pRsortBufferToatalSize_ / sizeof(half)];
    // ArithProgression(indexTensor, (half)0, (half)1, expertCount_); // 生成组索引0 1 2 ......
    PipeBarrier<PIPE_V>();
    ProposalConcat(sortTmpBuffer, xBiasTensor, expertCount_ / ONE_REPEAT_SORT_NUM, 4);
    ProposalConcat(sortTmpBuffer, indexTensor_, expertCount_ / ONE_REPEAT_SORT_NUM, 5);
    RpSort16(sortTmpBuffer2, sortTmpBuffer, expertCount_ / ONE_REPEAT_SORT_NUM);
    uint16_t elementLengths[4] = {16, 16, 16, 16};
    struct MrgSort4Info srcInfo(elementLengths, false, 3, 1);
    for (uint32_t index = 0; index < expertCount_ / ONE_REPEAT_SORT_NUM / 2; ++index) {
        PipeBarrier<PIPE_V>();
        uint32_t offset = index * 2 * ONE_REPEAT_SORT_NUM * regionProposalByte_ / sizeof(half);
        AscendC::MrgSortSrcList<half> srcList;
        srcList.src1 = sortTmpBuffer2[offset];
        srcList.src2 = sortTmpBuffer2[offset + ONE_REPEAT_SORT_NUM * regionProposalByte_ / sizeof(half)];
        MrgSort4(sortedInGroupTensor[offset], srcList, srcInfo);
    }
    PipeBarrier<PIPE_V>();
    sortedInGroupQueue_.EnQue<half>(sortedInGroupTensor);
    xSigmoidAddBiasQueue_.FreeTensor(xBiasTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKGroupIndex()
{
    LocalTensor<half> sortedInGroupTensor = sortedInGroupQueue_.DeQue<half>();
    LocalTensor<half> sortedGroupTensor = sortedGroupQueue_.AllocTensor<half>();
    LocalTensor<half> gropedSortedScore = sharedTmpBuffer_.template ReinterpretCast<half>();
    LocalTensor<half> top2ScoreSumPerGroup = sharedTmpBuffer_[regionProposalByte_ * expertCount_].template ReinterpretCast<half>();
    LocalTensor<half> sortOutGroupBuffer = sharedTmpBuffer_[regionProposalByte_ * expertCount_ + ONE_REPEAT_SORT_NUM * regionProposalByte_].template ReinterpretCast<half>();
    PipeBarrier<PIPE_V>();
    ProposalExtract(gropedSortedScore, sortedInGroupTensor, expertCount_ / ONE_REPEAT_SORT_NUM, 4);

    // GetValue替代GatherMask
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint32_t i = 0; i < expertCount_ / (ONE_REPEAT_SORT_NUM * 2); ++i) {
        top2ScoreSumPerGroup.SetValue(2 * i, gropedSortedScore.GetValue(i * 32));
        top2ScoreSumPerGroup.SetValue(2 * i + 1, gropedSortedScore.GetValue(i * 32 + 1));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    PipeBarrier<PIPE_V>();
    PairReduceSum(top2ScoreSumPerGroup, top2ScoreSumPerGroup, 1, groupCount_ * 2, 1, 1,
                  1); // 计算每个组内最大的两个数之和

    // 用最小值补到16个数
    int64_t duplicateNum = ONE_REPEAT_SORT_NUM - groupCount_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << groupCount_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(top2ScoreSumPerGroup, MIN_FP16, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    // 排序，将kgroup选出来
    ProposalConcat(sortOutGroupBuffer, top2ScoreSumPerGroup, 1, 4);
    ProposalConcat(sortOutGroupBuffer, indexTensor_, 1, 5);
    RpSort16(sortedGroupTensor, sortOutGroupBuffer, 1);
    PipeBarrier<PIPE_V>();

    ProposalExtract(top2ScoreSumPerGroup, sortedGroupTensor, 1, 5);

    PipeBarrier<PIPE_V>();
    duplicateNum = ONE_REPEAT_SORT_NUM - kGroup_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << kGroup_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(top2ScoreSumPerGroup, MIN_FP16, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    ProposalConcat(sortOutGroupBuffer, top2ScoreSumPerGroup, 1, 4);
    RpSort16(sortedGroupTensor, sortOutGroupBuffer, 1);

    sortedInGroupQueue_.EnQue<half>(sortedInGroupTensor);
    sortedGroupQueue_.EnQue<half>(sortedGroupTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKExpertIdx()
{
    LocalTensor<half> sortedInGroupTensor = sortedInGroupQueue_.DeQue<half>();
    LocalTensor<half> sortedGroupTensor = sortedGroupQueue_.DeQue<half>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<half> topKGroupIndexTensor = sharedTmpBuffer_.template ReinterpretCast<half>();
    PipeBarrier<PIPE_V>();
    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 4);

    LocalTensor<int32_t> topKGroupIndexTensorInt = topKGroupIndexTensor[ONE_REPEAT_SORT_NUM * regionProposalByte_ / sizeof(half)].template ReinterpretCast<int32_t>();
    Cast(topKGroupIndexTensorInt, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = k_;
    params.elementLengths[1] = k_;
    params.elementLengths[2] = k_;
    params.elementLengths[3] = k_;
    params.ifExhaustedSuspension = true;
    params.validBit = 0b1111;
    params.repeatTimes = 1;
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int64_t listOffset1 = topKGroupIndexTensorInt.GetValue(3) * perGroupExpertCount_ * 8;
    int64_t listOffset2 = topKGroupIndexTensorInt.GetValue(2) * perGroupExpertCount_ * 8;
    int64_t listOffset3 = topKGroupIndexTensorInt.GetValue(1) * perGroupExpertCount_ * 8;
    int64_t listOffset4 = topKGroupIndexTensorInt.GetValue(0) * perGroupExpertCount_ * 8;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    AscendC::MrgSortSrcList<half> srcList;
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];

    MrgSort4<half>(sortedGroupTensor, srcList, params);

    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);
    Cast(expertIdxTensor, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);
    PipeBarrier<PIPE_V>();
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    sortedGroupQueue_.FreeTensor(sortedGroupTensor);
    sortedInGroupQueue_.FreeTensor(sortedInGroupTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKExpertIdxEight()
{
    LocalTensor<half> sortedInGroupTensor = sortedInGroupQueue_.DeQue<half>();
    LocalTensor<half> sortedGroupTensor = sortedGroupQueue_.DeQue<half>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<half> topKGroupIndexTensor = sharedTmpBuffer_.template ReinterpretCast<half>();
    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 4);

    LocalTensor<int32_t> topKGroupIndexTensorInt = topKGroupIndexTensor[ONE_REPEAT_SORT_NUM * regionProposalByte_ / sizeof(half)].template ReinterpretCast<int32_t>();
    LocalTensor<half> finalSortTemp = topKGroupIndexTensorInt[ONE_REPEAT_SORT_NUM * regionProposalByte_].template ReinterpretCast<half>();
    Cast(topKGroupIndexTensorInt, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);

    // 0 ~ 3
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = k_;
    params.elementLengths[1] = k_;
    params.elementLengths[2] = k_;
    params.elementLengths[3] = k_;
    params.ifExhaustedSuspension = true;
    params.validBit = 0b1111;
    params.repeatTimes = 1;
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int64_t listOffset1 = topKGroupIndexTensorInt.GetValue(7) * perGroupExpertCount_ * 8;
    int64_t listOffset2 = topKGroupIndexTensorInt.GetValue(6) * perGroupExpertCount_ * 8;
    int64_t listOffset3 = topKGroupIndexTensorInt.GetValue(5) * perGroupExpertCount_ * 8;
    int64_t listOffset4 = topKGroupIndexTensorInt.GetValue(4) * perGroupExpertCount_ * 8;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    AscendC::MrgSortSrcList<half> srcList;
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];
    PipeBarrier<PIPE_V>();
    MrgSort4<half>(finalSortTemp, srcList, params);

    // 4 ~ 7
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    listOffset1 = topKGroupIndexTensorInt.GetValue(3) * perGroupExpertCount_ * 8;
    listOffset2 = topKGroupIndexTensorInt.GetValue(2) * perGroupExpertCount_ * 8;
    listOffset3 = topKGroupIndexTensorInt.GetValue(1) * perGroupExpertCount_ * 8;
    listOffset4 = topKGroupIndexTensorInt.GetValue(0) * perGroupExpertCount_ * 8;
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];

    MrgSort4<half>(finalSortTemp[4 * perGroupExpertCount_ * 8], srcList, params);
    // 归并
    params.elementLengths[0] = k_ * 4;
    params.elementLengths[1] = k_ * 4;
    params.elementLengths[2] = k_ * 4;
    params.elementLengths[3] = k_ * 4;
    params.ifExhaustedSuspension = true;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    srcList.src1 = finalSortTemp;
    srcList.src2 = finalSortTemp[4 * perGroupExpertCount_ * 8];

    MrgSort4<half>(sortedGroupTensor, srcList, params);
    PipeBarrier<PIPE_V>();
    // end
    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);
    Cast(expertIdxTensor, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);

    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    sortedGroupQueue_.FreeTensor(sortedGroupTensor);
    sortedInGroupQueue_.FreeTensor(sortedInGroupTensor);
}


template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKExpertScore()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<half> xSigmoidTensor = xSigmoidQueue_.DeQue<half>();
    LocalTensor<float> xSigmoidTensorFp32 = xSigmoidTensor[expertCountAlign_].template ReinterpretCast<float>();
    LocalTensor<half> yTensor = yOutQueue_.AllocTensor<half>();

    
    LocalTensor<float> reduceSumFp32Buffer = sharedTmpBuffer_.template ReinterpretCast<float>();

    // LocalTensor<float> reduceSumFp32Buffer = reduceSumBuffer_.template ReinterpretCast<float>();
    LocalTensor<float> yTensorFp32 = reduceSumFp32Buffer[(k_ + 63) / 64 * 64].template ReinterpretCast<float>();
    PipeBarrier<PIPE_V>();
    GatherV100(yTensorFp32, xSigmoidTensorFp32, expertIdxTensor, sharedTmpBuffer_, k_);
    PipeBarrier<PIPE_V>();
    ReduceSumFp32V100(reduceSumFp32Buffer, yTensorFp32, k_);

    Adds(reduceSumFp32Buffer, reduceSumFp32Buffer, eps_, 1);

    Duplicate(reduceSumFp32Buffer, reduceSumFp32Buffer.GetValue(0), k_);
    Div(yTensorFp32, yTensorFp32, reduceSumFp32Buffer, k_);
    Duplicate(reduceSumFp32Buffer, routedScalingFactor_, k_);
    Mul(yTensorFp32, yTensorFp32, reduceSumFp32Buffer, k_);
    Duplicate(yTensor, (half)0, (k_ + 15) / 16 * 16);
    Cast(yTensor, yTensorFp32, RoundMode::CAST_NONE, k_);
    PipeBarrier<PIPE_V>();
    xSigmoidQueue_.EnQue<half>(xSigmoidTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    yOutQueue_.EnQue<half>(yTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyOut(int64_t row)
{
    LocalTensor<half> yOutTensor = yOutQueue_.DeQue<half>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<half> xSigmoidTensor = xSigmoidQueue_.DeQue<half>();
    LocalTensor<float> yOutTensorFp32 = yOutTensor.template ReinterpretCast<float>();
    SetAtomicAdd<float>();
    DataCopy(yGmFp32_[row * k_ / FakerFp32RealFp16Multi],
                      yOutTensorFp32,
                      ((k_ / FakerFp32RealFp16Multi) + 7) / 8 * 8);
    SetAtomicNone();
    DataCopy(expertIdxGm_[row * k_], expertIdxTensor, (k_ + 7) / 8 * 8);
    xSigmoidQueue_.FreeTensor(xSigmoidTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxTensor);
    yOutQueue_.FreeTensor(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                        GM_ADDR out, GM_ADDR workspace,
                                                        const MoeGatingTopKTilingData *tilingData, TPipe *tPipe)
{
    syncLen_ = GetBlockNum() * DEFAULT_SYNCALL_NEED_SIZE;
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    rowCount_ = tilingData_->rowCount;
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    addBias_ = tilingData_->addBias == 1;
    k_ = tilingData_->k;
    kGroup_ = tilingData_->kGroup;
    groupCount_ = tilingData_->groupCount;
    perGroupExpertCount_ = tilingData_->perGroupExpertCount;
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;

    expertCountAlign_ = Align(expertCount_, 16);
    kAlign_ = Align(expertCount_, sizeof(float));

    // init output gm buf
    // bias
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);

    yGmFp32_.SetGlobalBuffer((__gm__ float *)y + (perCoreRowCount_ * k_ * blockIdx_) / FakerFp32RealFp16Multi, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ T *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    pipe_->InitBuffer(syncTmpSpaceBuffer_, syncLen_ * sizeof(int32_t));
    pipe_->InitBuffer(biasInBuffer_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(xInQueue_, 2, expertCountAlign_ * sizeof(T));
    pipe_->InitBuffer(xSigmoidQueue_, 1, expertCountAlign_ * (sizeof(float) + sizeof(T)));
    pipe_->InitBuffer(xSigmoidAddBiasQueue_, 1, expertCountAlign_ * sizeof(float));

    pRsortBufferToatalSize_ = expertCount_ * regionProposalByte_;
    pRsoetExBuffer_ = 2 * regionProposalByte_;
    pipe_->InitBuffer(sortGroupTmpBuffer_, pRsortBufferToatalSize_ * 2 + pRsoetExBuffer_);

    // 组排序结果
    pipe_->InitBuffer(sortedInGroupQueue_, 1, pRsortBufferToatalSize_);
    pipe_->InitBuffer(yOutQueue_, 2, kAlign_ * sizeof(T));
    pipe_->InitBuffer(expertIdxOutQueue_, 2, AlignBytes(k_, sizeof(int32_t)));
    pipe_->InitBuffer(outOutQueue_, 2, AlignBytes(expertCount_, sizeof(float)));
    pipe_->InitBuffer(sortedGroupQueue_, 1, regionProposalByte_ * expertCount_);
    pipe_->InitBuffer(calcTmpBuffer_, 128 * 1024);

    indexTensor_ = calcTmpBuffer_.Get<half>();
    sharedTmpBuffer_ = indexTensor_[expertCount_].template ReinterpretCast<uint8_t>();

    // sync
    syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace, syncLen_);

    LocalTensor<int32_t> syncLocal = syncTmpSpaceBuffer_.Get<int32_t>();
    Duplicate<int32_t>(syncLocal, (int32_t)0, syncLen_);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopy(syncTmpSpaceGm_, syncLocal, syncLen_);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::Process()
{

    ClearGMForAtomicAdd();

    LocalTensor<int32_t> syncLocal = syncTmpSpaceBuffer_.Get<int32_t>();
    AscendC::SyncAll(syncTmpSpaceGm_, syncLocal, GetBlockNum());
    PipeBarrier<PIPE_V>();
    ArithProgression(indexTensor_, (half)0, (half)1, expertCount_); // 生成组索引
    PipeBarrier<PIPE_V>();
    CopyInBias();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        SortInGroup();
        SelectTopKGroupIndex();
        if (kGroup_ == 4) {
            SelectTopKExpertIdx();
        } else {
            SelectTopKExpertIdxEight();
        }
        SelectTopKExpertScore();
        CopyOut(row);
    }

}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_FULLLOAD_H