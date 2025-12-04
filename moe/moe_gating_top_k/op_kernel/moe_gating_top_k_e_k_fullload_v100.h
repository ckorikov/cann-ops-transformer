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

template <typename T>
__aicore__ inline void GatherV100(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                  const LocalTensor<uint32_t>& indexTensor,
                                  const uint32_t count) {
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t realIndex = indexTensor.GetValue(index);
        dst.SetValue(index, src.GetValue(realIndex));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template <typename T>
__aicore__ inline void ReduceSumFp32V100(const LocalTensor<T>& dst,
                                         const LocalTensor<T>& src, const uint32_t count) {
    Duplicate(dst, (T)0, 64);
    Add(dst, dst, src, 64, count / 64, {1, 1, 1, 0, 0, 8});
    for (uint32_t index = 32; index >= 8; index = index / 2) {
        Add(dst, dst, dst[index], index);
    }
}


template <typename T>
class MoeGatingTopKEKFullload {
public:
    __aicore__ inline MoeGatingTopKEKFullload(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
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

    LocalTensor<half> indexTensor_;
    LocalTensor<half> gropedSortedScore_;
    LocalTensor<half> top2ScoreSumPerGroup_;
    LocalTensor<half> sortOutGroupBuffer_;
    LocalTensor<half> reduceSumBuffer_;
    LocalTensor<uint8_t> sharedTmpBuffer_;

    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> outGm_;

    int64_t blockIdx_;
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
    uint32_t regionProposalSize_ = 16;

    const MoeGatingTopKTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyInBias()
{
    LocalTensor<half> biasTensor = biasInBuffer_.Get<half>();
    DataCopy(biasTensor, biasGm_, expertCount_);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyInX(int64_t row)
{
    LocalTensor<half> xInLocalTensor = xInQueue_.AllocTensor<half>();
    DataCopy(xInLocalTensor, xGm_[row * expertCount_], expertCount_);
    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::ComputeX()
{
    LocalTensor<half> xInLocalTensor = xInQueue_.DeQue<half>();
    LocalTensor<half> biasTensor = biasInBuffer_.Get<half>();
    LocalTensor<half> xSigmoidTensor = xSigmoidQueue_.AllocTensor<half>();
    LocalTensor<half> xBiasTensor = xSigmoidAddBiasQueue_.AllocTensor<half>();

    Sigmoid(xSigmoidTensor, xInLocalTensor, sharedTmpBuffer_, expertCount_);
    PipeBarrier<PIPE_V>();
    if (addBias_) {
        Add(xBiasTensor, xSigmoidTensor, biasTensor, expertCount_);
    } else {
        Adds(xBiasTensor, xSigmoidTensor, static_cast<half>(0), expertCount_);
    }

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
        uint32_t offset = index * 2 * ONE_REPEAT_SORT_NUM * regionProposalSize_ / sizeof(half);
        AscendC::MrgSortSrcList<half> srcList;
        srcList.src1 = sortTmpBuffer2[offset];
        srcList.src2 = sortTmpBuffer2[offset + ONE_REPEAT_SORT_NUM * regionProposalSize_ / sizeof(half)];
        MrgSort4(sortedInGroupTensor[offset], srcList, srcInfo);
    }
    PipeBarrier<PIPE_V>();
    sortedInGroupQueue_.EnQue<half>(sortedInGroupTensor);
    xSigmoidAddBiasQueue_.FreeTensor(xBiasTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKGroupIndex()
{

    // 这一步是把8组专家各自的top2 score拿出来 2 * 8共16个数，用ProposalExtract替代，用
    // LocalTensor<half> top2ValueInGroupTensor = calcTmpBuffer_[indexMax + expertCount_];
    LocalTensor<half> sortedInGroupTensor = sortedInGroupQueue_.DeQue<half>();
    LocalTensor<half> sortedGroupTensor = sortedGroupQueue_.AllocTensor<half>();
    ProposalExtract(gropedSortedScore_, sortedInGroupTensor, expertCount_ / ONE_REPEAT_SORT_NUM, 4);

    // GetValue替代GatherMask
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint32_t i = 0; i < expertCount_ / (ONE_REPEAT_SORT_NUM * 2); ++i) {
        top2ScoreSumPerGroup_.SetValue(2 * i, gropedSortedScore_.GetValue(i * 32));
        top2ScoreSumPerGroup_.SetValue(2 * i + 1, gropedSortedScore_.GetValue(i * 32 + 1));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    PipeBarrier<PIPE_V>();

    PairReduceSum(top2ScoreSumPerGroup_, top2ScoreSumPerGroup_, 1, groupCount_ * 2, 1, 1,
                  1); // 计算每个组内最大的两个数之和

    PipeBarrier<PIPE_V>();
    // 用最小值补到16个数
    int64_t duplicateNum = ONE_REPEAT_SORT_NUM - groupCount_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << groupCount_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(top2ScoreSumPerGroup_, MIN_FP16, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    // 排序，将kgroup选出来

    ProposalConcat(sortOutGroupBuffer_, top2ScoreSumPerGroup_, 1, 4);
    ProposalConcat(sortOutGroupBuffer_, indexTensor_, 1, 5);
    RpSort16(sortedGroupTensor, sortOutGroupBuffer_, 1);

    // // ----------temp
    // PipeBarrier<PIPE_ALL>();
    // LocalTensor<half> topKGroupIndexTensor = sharedTmpBuffer_.template ReinterpretCast<half>();
    // ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 4);
    // DataCopy(yGm_, topKGroupIndexTensor, 16);
    // PipeBarrier<PIPE_ALL>();

    // ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);
    // LocalTensor<int32_t> topKGroupIndexTensorInt = topKGroupIndexTensor[ONE_REPEAT_SORT_NUM].template ReinterpretCast<int32_t>();
    // Cast(topKGroupIndexTensorInt, topKGroupIndexTensor, RoundMode::CAST_ROUND, 16);
    // DataCopy(expertIdxGm_, topKGroupIndexTensorInt, 16);
    // PipeBarrier<PIPE_ALL>();
    // // ----------temp
    // PipeBarrier<PIPE_V>();
    // LocalTensor<half> sortedGroupIndexTensor = indexTensor;
    // ProposalExtract(sortedGroupIndexTensor, sortedGroupTensor, 1, 5);

    // 以下代码是不是不需要 TODO
    // 需要将组排序(这里是降序，所以下mrgsor的时候反着取，3、2、1、0)
    // Cast(sortedGroupTensor, sortedGroupIndexTensor, RoundMode::CAST_ROUND, kGroup_);
    // PipeBarrier<PIPE_V>();
    // duplicateNum = ONE_REPEAT_SORT_NUM - kGroup_;
    // if (duplicateNum > 0) {
    //     uint64_t mask0 = UINT64_MAX << kGroup_;
    //     uint64_t mask[2] = {mask0, 0};
    //     Duplicate(sortedGroupTensor, MIN_FP16, mask, 1, 1, 8);
    //     PipeBarrier<PIPE_V>();
    // }

    // ProposalConcat(sortTmpBuffer, sortedGroupTensor, 4, 4);
    // ProposalConcat(sortTmpBuffer, sortedGroupIndexTensor, 4, 5);
    // RpSort16(top2ValueInGroupTensor, sortTmpBuffer, 1);
    // PipeBarrier<PIPE_V>();
    // ProposalExtract(sortedGroupTensor, top2ValueInGroupTensor, 1, 4);

    // PipeBarrier<PIPE_V>();
    // Cast(sortedGroupIndexTensor, sortedGroupTensor, RoundMode::CAST_ROUND, kGroup_);
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
    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);

    LocalTensor<int32_t> topKGroupIndexTensorInt = topKGroupIndexTensor[ONE_REPEAT_SORT_NUM].template ReinterpretCast<int32_t>();
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
    int64_t listOffset1 = topKGroupIndexTensorInt.GetValue(0) * perGroupExpertCount_ * 8;
    int64_t listOffset2 = topKGroupIndexTensorInt.GetValue(1) * perGroupExpertCount_ * 8;
    int64_t listOffset3 = topKGroupIndexTensorInt.GetValue(2) * perGroupExpertCount_ * 8;
    int64_t listOffset4 = topKGroupIndexTensorInt.GetValue(3) * perGroupExpertCount_ * 8;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    AscendC::MrgSortSrcList<half> srcList;
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];

    MrgSort4<half>(sortedGroupTensor, srcList, params);
    PipeBarrier<PIPE_V>();
    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);
    Cast(expertIdxTensor, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);

    expertIdxOutQueue_.EnQue(expertIdxTensor);
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
    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);

    LocalTensor<int32_t> topKGroupIndexTensorInt = topKGroupIndexTensor[ONE_REPEAT_SORT_NUM].template ReinterpretCast<int32_t>();
    Cast(topKGroupIndexTensorInt, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);

    AscendC::MrgSort4Info params;
    params.elementLengths[0] = k_;
    params.elementLengths[1] = k_;
    params.elementLengths[2] = k_;
    params.elementLengths[3] = k_;
    params.ifExhaustedSuspension = true;
    params.validBit = 0b1111;
    params.repeatTimes = 2;
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int64_t listOffset1 = topKGroupIndexTensorInt.GetValue(0) * perGroupExpertCount_ * 8;
    int64_t listOffset2 = topKGroupIndexTensorInt.GetValue(1) * perGroupExpertCount_ * 8;
    int64_t listOffset3 = topKGroupIndexTensorInt.GetValue(2) * perGroupExpertCount_ * 8;
    int64_t listOffset4 = topKGroupIndexTensorInt.GetValue(3) * perGroupExpertCount_ * 8;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    AscendC::MrgSortSrcList<half> srcList;
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];

    MrgSort4<half>(sortedGroupTensor, srcList, params);
    PipeBarrier<PIPE_V>();


    ProposalExtract(topKGroupIndexTensor, sortedGroupTensor, 1, 5);
    Cast(expertIdxTensor, topKGroupIndexTensor, RoundMode::CAST_ROUND, k_);

    expertIdxOutQueue_.EnQue(expertIdxTensor);
    sortedGroupQueue_.FreeTensor(sortedGroupTensor);
    sortedInGroupQueue_.FreeTensor(sortedInGroupTensor);
}


template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKExpertScore()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<half> xSigmoidTensor = xSigmoidQueue_.DeQue<half>();
    LocalTensor<half> yTensor = yOutQueue_.AllocTensor<half>();

    // LocalTensor<half> calTensor = calcTmpBuffer_.Get<half>();

    GatherV100(yTensor, xSigmoidTensor, expertIdxTensor.template ReinterpretCast<uint32_t>(), k_);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceSumBuffer_, yTensor, sharedTmpBuffer_.template ReinterpretCast<half>(), k_);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    LocalTensor<float> reduceSumFp32Buffer = reduceSumBuffer_.template ReinterpretCast<float>();
    Cast(reduceSumFp32Buffer, reduceSumBuffer_, RoundMode::CAST_NONE, 1);
    Adds(reduceSumFp32Buffer, reduceSumFp32Buffer, eps_, 1);
    Cast(reduceSumBuffer_, reduceSumFp32Buffer, RoundMode::CAST_NONE, 1);
    // // temp
    // PipeBarrier<PIPE_V>();
    // DataCopy(yGm_, reduceSumBuffer_, 16);
    // PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    half sumValue = reduceSumBuffer_.GetValue(0);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Duplicate(reduceSumBuffer_, sumValue, k_);
    PipeBarrier<PIPE_V>();
    Div(yTensor, yTensor, reduceSumBuffer_, k_);
    // PipeBarrier<PIPE_V>();
    // DataCopy(yGm_, yTensor, 16);
    // PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_V>();

    Duplicate(reduceSumFp32Buffer, routedScalingFactor_, k_);
    Cast(reduceSumBuffer_, reduceSumFp32Buffer, RoundMode::CAST_NONE, k_);

    Mul(yTensor, yTensor, reduceSumBuffer_, k_);

    xSigmoidQueue_.EnQue<half>(xSigmoidTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    yOutQueue_.EnQue(yTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyOut(int64_t row)
{
    LocalTensor<half> yOutTensor = yOutQueue_.DeQue<half>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<half> xSigmoidTensor = xSigmoidQueue_.DeQue<half>();
    uint32_t alignOut = (k_ + 15) / 16 * 16;
    DataCopy(yGm_[row * k_], yOutTensor, alignOut);
    DataCopy(expertIdxGm_[row * k_], expertIdxTensor, alignOut);
    xSigmoidQueue_.FreeTensor(xSigmoidTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxTensor);
    yOutQueue_.FreeTensor(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                        GM_ADDR out, GM_ADDR workspace,
                                                        const MoeGatingTopKTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    addBias_ = tilingData_->addBias == 1;
    k_ = tilingData_->k;
    kGroup_ = tilingData_->kGroup; // 32
    groupCount_ = tilingData_->groupCount; // 8
    perGroupExpertCount_ = tilingData_->perGroupExpertCount; // 32
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;

    expertCountAlign_ = Align(expertCount_, sizeof(float));
    kAlign_ = Align(expertCount_, sizeof(float));

    // bias
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
    pipe_->InitBuffer(biasInBuffer_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));

    // x
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    pipe_->InitBuffer(xInQueue_, 2, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));

    pipe_->InitBuffer(xSigmoidQueue_, 1, AlignBytes(expertCount_, sizeof(float)));
    pipe_->InitBuffer(xSigmoidAddBiasQueue_, 1, AlignBytes(expertCount_, sizeof(float)));

    pipe_->InitBuffer(calcTmpBuffer_, tilingData_->calTmpBufUbSize);
    pRsortBufferToatalSize_ = expertCount_ * regionProposalSize_;
    pRsoetExBuffer_ = 2 * regionProposalSize_;
    pipe_->InitBuffer(sortGroupTmpBuffer_, pRsortBufferToatalSize_ * 2 + pRsoetExBuffer_);

    // 组排序结果
    pipe_->InitBuffer(sortedInGroupQueue_, 1, pRsortBufferToatalSize_);

    // init output gm buf
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ T *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);


    pipe_->InitBuffer(yOutQueue_, 2, kAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(expertIdxOutQueue_, 2, AlignBytes(k_, sizeof(int32_t)));
    pipe_->InitBuffer(outOutQueue_, 2, AlignBytes(expertCount_, sizeof(float)));

    pipe_->InitBuffer(sortedGroupQueue_, 1,
                      (groupCount_ + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM *
                          sizeof(float) * 2);

    indexTensor_ = calcTmpBuffer_.Get<half>();
    gropedSortedScore_ = indexTensor_[expertCount_];
    top2ScoreSumPerGroup_ = gropedSortedScore_[expertCount_];
    sortOutGroupBuffer_ = top2ScoreSumPerGroup_[expertCount_ / ONE_REPEAT_SORT_NUM];
    reduceSumBuffer_ = sortOutGroupBuffer_[ONE_REPEAT_SORT_NUM * regionProposalSize_ / sizeof(half)];
    sharedTmpBuffer_ = reduceSumBuffer_[(k_ + 15) / 16 * 16].template ReinterpretCast<uint8_t>();

    PipeBarrier<PIPE_V>();
    ArithProgression(indexTensor_, (half)0, (half)1, expertCount_); // 生成组索引
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::Process()
{
    CopyInBias();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        SortInGroup();
        SelectTopKGroupIndex();
        if (k_ == 4) {
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