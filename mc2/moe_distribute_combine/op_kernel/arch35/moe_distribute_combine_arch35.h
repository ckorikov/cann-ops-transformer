/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

/*!
 * \file moe_distribute_combine_a5.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A5_H
#define MOE_DISTRIBUTE_COMBINE_A5_H

#include "lib/hccl/hccl.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"


namespace MoeDistributeCombineA5Impl {
constexpr uint8_t BUFFER_NUM = 2;               // 多buf
constexpr uint32_t UB_ALIGN = 32;                 // UB按32字节对齐

constexpr uint32_t ALIGN_DOWN_TO_32_MASK = 31;
constexpr uint32_t NEED_THIRTY_FIRST = 31;
constexpr uint32_t RIGHT_SHIFT_BIT_FIVE = 5;
constexpr uint64_t HALF_DATA_DIV = 2;
constexpr uint32_t DUAL_DATA = 2;
constexpr uint16_t BLOCK_COUNT = 2;

template <typename T>
__aicore__ inline T Ceil32(T x)
{
    return (x + NEED_THIRTY_FIRST) >> RIGHT_SHIFT_BIT_FIVE;
}

template <typename T>
__aicore__ inline T Align32(T x)
{
    return (x + ALIGN_DOWN_TO_32_MASK) & (~ALIGN_DOWN_TO_32_MASK);
}

using namespace AscendC;

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

template <TemplateMC2TypeClass>
class MoeDistributeCombineA5 {
public:
    __aicore__ inline MoeDistributeCombineA5(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR sharedExpertX, GM_ADDR XOut, GM_ADDR workspaceGM,
                                TPipe *pipe, const MoeDistributeCombineTilingDataA5 *tilingData);
    __aicore__ inline void Process();
private:
    // 按照rank分核逻辑，得到每个核的起始和数量
    __aicore__ inline void SplitRank(uint32_t &start, uint32_t &count);
    // 准备发送数据的初始化
    __aicore__ inline void PrepareInit();
    // 数据复制与填充
    __aicore__ inline void CopyAndPadData(uint32_t &startRank, uint32_t &rankNum, uint32_t &eachSize,
                                          LocalTensor<uint64_t> &sizeLT, LocalTensor<uint64_t> &sendOffsetLT);
    // 共享专家卡通信准备
    __aicore__ inline void HandleSharedToken();
    __aicore__ inline void HandleSharedRank();
    __aicore__ inline void SharedPrepare();
    // MOE专家卡通信准备
    __aicore__ inline void HandleMoeRank(uint32_t &startRank, LocalTensor<uint64_t> &sizeLT, uint32_t &eachCnt,
                                         uint32_t &rankNum, LocalTensor<uint32_t> &offsetLT,
                                         LocalTensor<uint64_t> &sendOffsetLT);
    __aicore__ inline void MoePrepare();
    // 准备发送数据
    __aicore__ inline void PrepareSendData();
    // 通信
    __aicore__ inline void Communication();
    // 计算的初始化
    __aicore__ inline void CalculateInit();
    // 计算并输出
    __aicore__ inline void HandleCalculateToken(uint32_t &beginIndex, uint32_t &endIndex, uint32_t &processLen,
                                                uint32_t &tokenOffset, DataCopyExtParams &copyOutParams,
                                                GlobalTensor<ExpandXType> &rowTmpGT, GM_ADDR sharedBase);
    __aicore__ inline void Calculate();

    __aicore__ inline void HandleSharedExpertX(uint32_t &tokenIndex, uint32_t &tokenOffset, DataCopyParams &copyInParams,
                                               DataCopyPadParams &padInParams, uint32_t &processLen);
    TPipe *pipe_;

    GlobalTensor<ExpandXType> expandXGT_;
    GlobalTensor<ExpandIdxType> expertIdsGT_;
    GlobalTensor<ExpandIdxType> expandIdxGT_;
    GlobalTensor<float> expandScalesGT_;
    GlobalTensor<ExpandIdxType> epSendCountGT_;
    GlobalTensor<ExpandXType> expandOutGT_;

    LocalTensor<int32_t> inputCountLT_;
    LocalTensor<ExpandIdxType> expertIdsLT_;
    LocalTensor<ExpandIdxType> expandIdxLT_;
    LocalTensor<float> expandScalesLT_;
    LocalTensor<float> castLT_;
    LocalTensor<float> mulLT_;
    LocalTensor<float> sumLT_;

    uint32_t axisBS_;
    uint32_t axisH_;
    uint32_t axisK_;
    uint32_t aivNum_;
    uint32_t epWorldSize_;
    uint32_t epRankId_;
    uint32_t aivId_;
    uint32_t sharedExpertRankNum_;
    uint32_t moeExpertNum_;
    uint32_t moeExpertRankNum_;
    uint32_t localExpertNum_;
    uint32_t hasSharedExpertX_;

    uint32_t bskNum_;
    uint32_t perTokenSize_;
    uint32_t perRankDataSize_;
    bool isShareExpertRank_;

    TQue<QuePosition::VECIN, BUFFER_NUM> moeQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sharedQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sharedExpertXQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> tokenQue_;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
    AscendC::HcclHandle hcclHandleId_;

    GM_ADDR sendBufGM_;
    GM_ADDR sendSizeGM_;
    GM_ADDR sendOffsetGM_;
    GM_ADDR recvBufGM_;
    GM_ADDR sharedExpertXGM_;
    uint64_t recvOffset_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::Init(GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR sharedExpertX,
    GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineTilingDataA5 *tilingData)
{
    pipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->combineTilingInfo.epRankId;
    axisBS_ = tilingData->combineTilingInfo.bs;
    axisH_ = tilingData->combineTilingInfo.h;
    axisK_ = tilingData->combineTilingInfo.k;
    aivNum_ = tilingData->combineTilingInfo.aivNum;
    sharedExpertRankNum_ = tilingData->combineTilingInfo.sharedExpertRankNum;
    epWorldSize_ = tilingData->combineTilingInfo.epWorldSize;
    moeExpertNum_ = tilingData->combineTilingInfo.moeExpertNum;
    hasSharedExpertX_ = tilingData->combineTilingInfo.hasSharedExpertX;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    localExpertNum_ = moeExpertNum_ / moeExpertRankNum_;
    isShareExpertRank_ = epRankId_ < sharedExpertRankNum_;
    bskNum_ = axisBS_ * axisK_;
    perTokenSize_ = axisH_ * sizeof(ExpandXType);
    perRankDataSize_ = perTokenSize_ * axisBS_ * localExpertNum_;

    expandXGT_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGT_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandIdxGT_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    epSendCountGT_.SetGlobalBuffer((__gm__ int32_t *)epSendCount);
    expandScalesGT_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGT_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);

    sharedExpertXGM_ = sharedExpertX;
    sendBufGM_ = workspaceGM;
    sendSizeGM_ = sendBufGM_ + epWorldSize_ * perRankDataSize_;
    sendOffsetGM_ = sendSizeGM_ + epWorldSize_ * sizeof(uint64_t) * DUAL_DATA;

    GlobalTensor<int32_t> statusGT;
    __gm__ HcclCombineOpParam *context = (__gm__ HcclCombineOpParam *)(GetHcclContext<0>());
    hccl_.Init((GM_ADDR)context);
    GM_ADDR cclBuf = (GM_ADDR)context->windowsOut[0];
    uint64_t statusSize = aivNum_ * UB_ALIGN;
    statusGT.SetGlobalBuffer((__gm__ int32_t*)(cclBuf + aivId_ * UB_ALIGN));
    if (statusGT(0) == 0) {
        recvBufGM_ = cclBuf + statusSize;
        recvOffset_ = statusSize + epRankId_ * perRankDataSize_;
        statusGT(0) = 1;
    } else {
        uint64_t halfDataSize = (context->winSize - statusSize) / HALF_DATA_DIV;
        recvBufGM_ = cclBuf + statusSize + halfDataSize;
        recvOffset_ = statusSize + halfDataSize + epRankId_ * perRankDataSize_;
        statusGT(0) = 0;
    }

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGT);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::CopyAndPadData(uint32_t &startRank,
    uint32_t &rankNum, uint32_t &eachSize, LocalTensor<uint64_t> &sizeLT, LocalTensor<uint64_t> &sendOffsetLT)
{
    GlobalTensor<uint64_t> sendGT;
    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendSizeGM_ + startRank * sizeof(uint64_t)));
    DataCopyExtParams params = {BLOCK_COUNT, static_cast<uint32_t>(rankNum * sizeof(uint64_t)),
        static_cast<uint32_t>(Ceil32(eachSize) - Ceil32(rankNum * sizeof(uint64_t))),
        static_cast<uint32_t>((epWorldSize_ - rankNum) * sizeof(uint64_t)), 0};
    DataCopyPad(sendGT, sizeLT, params);

    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendOffsetGM_ + startRank * sizeof(uint64_t)));
    DataCopyPad(sendGT, sendOffsetLT, params);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::SplitRank(uint32_t &start, uint32_t &count)
{
    count = epWorldSize_ / aivNum_;
    uint32_t remainCnt = epWorldSize_ % aivNum_;
    start = count * aivId_;

    if (aivId_ < remainCnt) {
        ++count;
        start += aivId_;
    } else {
        start += remainCnt;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::PrepareInit()
{
    pipe_->InitBuffer(tokenQue_, BUFFER_NUM, perTokenSize_);
    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, localExpertNum_ * epWorldSize_ * sizeof(int32_t));
    inputCountLT_ = tmpBuf.Get<int32_t>();
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(localExpertNum_ * epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyPad(inputCountLT_, epSendCountGT_, copyParams, padParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::HandleSharedToken()
{
    uint32_t totalToken = (epWorldSize_ / sharedExpertRankNum_) * axisBS_ - axisBS_; // 本卡的不需要通过通信发送
    uint32_t tokenTile = totalToken / aivNum_;
    uint32_t remainTokTile = totalToken % aivNum_;
    uint32_t startToken = tokenTile * aivId_;
    if (aivId_ < remainTokTile) {
        tokenTile++;
        startToken = tokenTile * aivId_;
    } else {
        startToken += remainTokTile;
    }
    uint32_t endToken = startToken + tokenTile;
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(perTokenSize_), 0U, 0U};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    GlobalTensor<ExpandXType> tokenGT;

    for (uint32_t i = startToken; i < endToken; ++i) {
        uint32_t tokenIdx = i + axisBS_;
        uint32_t toRankId = tokenIdx / axisBS_ * sharedExpertRankNum_ + epRankId_;
        auto t = tokenQue_.AllocTensor<ExpandXType>();
        DataCopyPad(t, expandXGT_[tokenIdx * axisH_], copyParams, padParams);
        tokenQue_.EnQue(t);
        t = tokenQue_.DeQue<ExpandXType>();
        GM_ADDR dstGM = sendBufGM_ + toRankId * perRankDataSize_ + (tokenIdx % axisBS_) * perTokenSize_;
        tokenGT.SetGlobalBuffer((__gm__ ExpandXType*)dstGM);
        DataCopyPad(tokenGT, t, copyParams);
        tokenQue_.FreeTensor<ExpandXType>(t);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::HandleSharedRank()
{
    uint32_t startRank;
    uint32_t rankNum;
    SplitRank(startRank, rankNum);
    if (rankNum == 0) {
        return;
    }
    uint32_t endRank = startRank + rankNum;
    TBuf<> sizeBuf;
    uint32_t eachSize = Align32(rankNum * sizeof(uint64_t));
    uint32_t secondOffset = eachSize / sizeof(uint64_t);
    pipe_->InitBuffer(sizeBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sizeLT = sizeBuf.Get<uint64_t>();
    pipe_->InitBuffer(sizeBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sendOffsetLT = sizeBuf.Get<uint64_t>();

    for (uint32_t i = startRank; i < endRank; ++i) {
        uint64_t count = ((i >= sharedExpertRankNum_) && (i % sharedExpertRankNum_ == epRankId_)) ? axisBS_ : 0;
        uint64_t countByte = count * perTokenSize_;
        uint64_t halfSize = countByte / HALF_DATA_DIV;
        uint32_t idx = i - startRank;
        sizeLT(idx) = halfSize;
        sizeLT(idx + secondOffset) = countByte - halfSize;
        sendOffsetLT(idx) = i * perRankDataSize_;
        sendOffsetLT(idx + secondOffset) = i * perRankDataSize_ + halfSize;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    CopyAndPadData(startRank, rankNum, eachSize, sizeLT, sendOffsetLT);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::SharedPrepare()
{
    HandleSharedToken();
    HandleSharedRank();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::HandleMoeRank(uint32_t &startRank,
    LocalTensor<uint64_t> &sizeLT, uint32_t &eachCnt, uint32_t &rankNum, LocalTensor<uint32_t> &offsetLT,
    LocalTensor<uint64_t> &sendOffsetLT)
{
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(perTokenSize_), 0U, 0U};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    GlobalTensor<ExpandXType> tokenGT;
    uint32_t inPreExpertCount = 0;
    uint32_t outPreExpertCount = 0;
    for (uint32_t k = 0; k < localExpertNum_; ++k) {
        uint32_t expertOffset = k * epWorldSize_;
        uint32_t inPreCount = startRank > 0 ? inputCountLT_(expertOffset + startRank - 1) : 0;
        for (uint32_t i = 0; i < rankNum; ++i) {
            uint32_t curRankId = startRank + i;
            uint32_t curSumNum = inputCountLT_(expertOffset + curRankId);
            uint32_t curTokenNum = curSumNum - inPreCount;
            uint32_t tokenBeginIdx = (inPreExpertCount + inPreCount) * axisH_;

            GM_ADDR dstGM = sendBufGM_ + curRankId * perRankDataSize_ + offsetLT(i) * perTokenSize_;
            for (uint32_t j = 0; j < curTokenNum; ++j) {
                auto t = tokenQue_.AllocTensor<ExpandXType>();
                DataCopyPad(t, expandXGT_[tokenBeginIdx], copyParams, padParams);
                tokenQue_.EnQue(t);
                t = tokenQue_.DeQue<ExpandXType>();
                tokenGT.SetGlobalBuffer((__gm__ ExpandXType*)dstGM);
                DataCopyPad(tokenGT, t, copyParams);
                tokenQue_.FreeTensor<ExpandXType>(t);
                tokenBeginIdx += axisH_;
                dstGM += perTokenSize_;
            }

            offsetLT(i) += curTokenNum;

            inPreCount = curSumNum;
            uint64_t countByte = curTokenNum * perTokenSize_;
            uint64_t halfSize = countByte / HALF_DATA_DIV;
            sizeLT(i) += halfSize;
            sizeLT(i + eachCnt) += countByte - halfSize;

            sendOffsetLT(i) = curRankId * perRankDataSize_;
            sendOffsetLT(i + eachCnt) = curRankId * perRankDataSize_ + halfSize;
        }

        inPreExpertCount += inputCountLT_((k + 1) * epWorldSize_ - 1);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::MoePrepare()
{
    uint32_t startRank;
    uint32_t rankNum;
    SplitRank(startRank, rankNum);
    if (rankNum == 0) {
        return;
    }
    uint32_t endRank = startRank + rankNum;
    uint32_t eachSize = Align32(rankNum * sizeof(uint64_t));
    uint32_t eachCnt = eachSize / sizeof(uint64_t);
    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, eachSize * DUAL_DATA + rankNum * sizeof(uint32_t));
    LocalTensor<uint64_t> sizeLT = tmpBuf.GetWithOffset<uint64_t>(eachCnt * DUAL_DATA, 0);
    LocalTensor<uint32_t> offsetLT = tmpBuf.GetWithOffset<uint32_t>(rankNum, eachSize * DUAL_DATA);
    pipe_->InitBuffer(tmpBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sendOffsetLT = tmpBuf.Get<uint64_t>();
    uint32_t rankCntSize = Align32(localExpertNum_ * sizeof(uint32_t));
    uint32_t rankCnt = rankCntSize / sizeof(uint32_t);
    LocalTensor<int32_t> sizeI32LT = sizeLT.ReinterpretCast<int32_t>();
    Duplicate(sizeI32LT, 0, DUAL_DATA * eachCnt * (sizeof(uint64_t) / sizeof(int32_t)) + rankNum);

    SyncFunc<AscendC::HardEvent::V_S>();

    HandleMoeRank(startRank, sizeLT, eachCnt, rankNum, offsetLT, sendOffsetLT);

    SyncFunc<AscendC::HardEvent::S_MTE3>();

    CopyAndPadData(startRank, rankNum, eachSize, sizeLT, sendOffsetLT);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::PrepareSendData()
{
    PrepareInit();
    if (isShareExpertRank_) {
        SharedPrepare();
    } else {
        MoePrepare();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::CalculateInit()
{
    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, bskNum_ * sizeof(int32_t));
    expertIdsLT_ = tmpBuf.Get<int32_t>();
    pipe_->InitBuffer(tmpBuf, bskNum_ * sizeof(ExpandIdxType));
    expandIdxLT_ = tmpBuf.Get<int32_t>();
    pipe_->InitBuffer(tmpBuf, bskNum_ * sizeof(float));
    expandScalesLT_ = tmpBuf.Get<float>();

    DataCopyExtParams bskParams = {1U, static_cast<uint32_t>(bskNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};

    DataCopyPad(expandIdxLT_, expandIdxGT_, bskParams, copyPadParams);
    DataCopyPad(expertIdsLT_, expertIdsGT_, bskParams, copyPadParams);
    DataCopyPad(expandScalesLT_, expandScalesGT_, bskParams, copyPadFloatParams);
    pipe_->InitBuffer(moeQue_, BUFFER_NUM, perTokenSize_);
    pipe_->InitBuffer(outQue_, BUFFER_NUM, perTokenSize_);
    if (sharedExpertRankNum_ > 0) {
        pipe_->InitBuffer(sharedQue_, BUFFER_NUM, perTokenSize_);
    }

    if (hasSharedExpertX_ > 0) {
        pipe_->InitBuffer(sharedExpertXQue_, BUFFER_NUM, perTokenSize_);
    }

    uint32_t tokenF32Size = axisH_ * sizeof(float);
    pipe_->InitBuffer(tmpBuf, tokenF32Size);
    castLT_ = tmpBuf.Get<float>();
    pipe_->InitBuffer(tmpBuf, tokenF32Size);
    mulLT_ = tmpBuf.Get<float>();
    pipe_->InitBuffer(tmpBuf, tokenF32Size);
    sumLT_ = tmpBuf.Get<float>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::Communication()
{
    SyncAll<true>();
    if (aivId_ == 0) {
        uint64_t localDataSize = 0;
        if (!isShareExpertRank_) {
            for (int i = 0; i < localExpertNum_; ++i) {
                int32_t idx = i * epWorldSize_ + epRankId_;
                int32_t count = epRankId_ > 0 ? (inputCountLT_(idx) - inputCountLT_(idx - 1)) : inputCountLT_(idx);
                localDataSize += count * perTokenSize_;
            }
        }

        hcclHandleId_ = hccl_.AlltoAllvWrite<true>(sendBufGM_, sendOffsetGM_, sendSizeGM_, recvOffset_, localDataSize);
    }

    CalculateInit(); // 在这里可以被通信开销掩盖

    if (aivId_ == 0) {
        hccl_.Wait(hcclHandleId_);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::HandleSharedExpertX(uint32_t &tokenIndex,
    uint32_t &tokenOffset, DataCopyParams &copyInParams, DataCopyPadParams &padInParams, uint32_t &processLen)
{
    GlobalTensor<ExpandXType> sharedExpertXGT;
    if (hasSharedExpertX_ > 0) {
        auto rowTmpLocal = sharedExpertXQue_.AllocTensor<ExpandXType>();
        GM_ADDR shareExpertXAddr = sharedExpertXGM_ + tokenIndex * perTokenSize_ + tokenOffset * sizeof(ExpandXType);
        sharedExpertXGT.SetGlobalBuffer((__gm__ ExpandXType *)(shareExpertXAddr));
        DataCopyPad(rowTmpLocal, sharedExpertXGT, copyInParams, padInParams);
        sharedExpertXQue_.EnQue(rowTmpLocal);
        rowTmpLocal = sharedExpertXQue_.DeQue<ExpandXType>();
        Cast(castLT_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
        AscendC::Add(sumLT_, sumLT_, castLT_, processLen);
        sharedExpertXQue_.FreeTensor<ExpandXType>(rowTmpLocal);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::HandleCalculateToken(uint32_t &beginIndex,
    uint32_t &endIndex, uint32_t &processLen, uint32_t &tokenOffset, DataCopyExtParams &copyOutParams,
    GlobalTensor<ExpandXType> &rowTmpGT, GM_ADDR sharedBase)
{
    DataCopyParams copyInParams = {1U, static_cast<uint16_t>(processLen * sizeof(ExpandXType)), 0U, 0U};
    DataCopyPadParams padInParams = {false, 0, 0, 0};
    for (uint32_t tokenIndex = beginIndex; tokenIndex < endIndex; tokenIndex++) {
        uint32_t index = tokenIndex * axisK_;
        Duplicate(sumLT_, (float)0, axisH_);
        for (uint32_t i = 0; i < axisK_; i++) {
            int32_t moeExpert = expertIdsLT_(index);
            int32_t expertRank = moeExpert / localExpertNum_;
            int32_t dataSrcRank = expertRank + sharedExpertRankNum_;
            int32_t inPreCount = expandIdxLT_(index);
            GM_ADDR src = recvBufGM_ + dataSrcRank * perRankDataSize_ + inPreCount * perTokenSize_ +
                          tokenOffset * sizeof(ExpandXType);

            rowTmpGT.SetGlobalBuffer((__gm__ ExpandXType *)src);
            float scaleVal = expandScalesLT_(index);
            auto t0 = moeQue_.AllocTensor<ExpandXType>();
            DataCopyPad(t0, rowTmpGT, copyInParams, padInParams);
            moeQue_.EnQue(t0);

            t0 = moeQue_.DeQue<ExpandXType>();
            Cast(castLT_, t0, AscendC::RoundMode::CAST_NONE, processLen);
            AscendC::Muls(mulLT_, castLT_, scaleVal, processLen);
            AscendC::Add(sumLT_, sumLT_, mulLT_, processLen);
            moeQue_.FreeTensor<ExpandXType>(t0);
            ++index;
        }

        if (sharedExpertRankNum_ > 0U) {
            auto t0 = sharedQue_.AllocTensor<ExpandXType>();
            GM_ADDR shareAddr = sharedBase + tokenIndex * perTokenSize_ + tokenOffset * sizeof(ExpandXType);
            rowTmpGT.SetGlobalBuffer((__gm__ ExpandXType *)(shareAddr));
            DataCopyPad(t0, rowTmpGT, copyInParams, padInParams);
            sharedQue_.EnQue(t0);
            t0 = sharedQue_.DeQue<ExpandXType>();
            Cast(castLT_, t0, AscendC::RoundMode::CAST_NONE, processLen);
            AscendC::Add(sumLT_, sumLT_, castLT_, processLen);
            sharedQue_.FreeTensor(t0);
        }

        HandleSharedExpertX(tokenIndex, tokenOffset, copyInParams, padInParams, processLen);
        auto outLT = outQue_.AllocTensor<ExpandXType>();
        Cast(outLT, sumLT_, AscendC::RoundMode::CAST_RINT, processLen);
        outQue_.EnQue(outLT);
        outLT = outQue_.DeQue<ExpandXType>();
        DataCopyPad(expandOutGT_[tokenIndex * axisH_ + tokenOffset], outLT, copyOutParams);
        outQue_.FreeTensor(outLT);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::Calculate()
{
    uint32_t beginIndex;
    uint32_t endIndex;
    uint32_t processLen;
    uint32_t tokenOffset;
    if (axisBS_ < aivNum_) {
        uint32_t aivNumPerToken = aivNum_ / axisBS_;
        if (aivId_ >= (axisBS_ * aivNumPerToken)) {
            return;
        }
        uint32_t tokenIndex = aivId_ / aivNumPerToken;
        uint32_t blockCnt = UB_ALIGN / sizeof(ExpandXType);
        processLen = (axisH_ / aivNumPerToken / blockCnt) * blockCnt;

        tokenOffset = processLen * (aivId_ % aivNumPerToken);
        if ((aivId_ % aivNumPerToken) == (aivNumPerToken - 1)) {
            processLen = axisH_ - ((aivNumPerToken - 1) * processLen);
        }
        beginIndex = tokenIndex;
        endIndex = beginIndex + 1;
    } else {
        uint32_t tokenPerAivNum = axisBS_ / aivNum_;
        uint32_t remainToken = axisBS_ % aivNum_;
        beginIndex = tokenPerAivNum * aivId_;
        if (aivId_ < remainToken) {
            tokenPerAivNum++;
            beginIndex = tokenPerAivNum * aivId_;
        } else {
            beginIndex += remainToken;
        }
        endIndex = beginIndex + tokenPerAivNum;
        processLen = axisH_;
        tokenOffset = 0;
    }
    if (processLen == 0) {
        return;
    }

    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(processLen * sizeof(ExpandXType)), 0, 0, 0};
    GlobalTensor<ExpandXType> rowTmpGT;

    GM_ADDR sharedBase = nullptr;
    if (sharedExpertRankNum_ > 0U) {
        uint32_t moeOnShareRank = epRankId_ % sharedExpertRankNum_;
        sharedBase = isShareExpertRank_ ? ((GM_ADDR)expandXGT_.GetPhyAddr()) :
                                          (recvBufGM_ + moeOnShareRank * perRankDataSize_);
    }

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    HandleCalculateToken(beginIndex, endIndex, processLen, tokenOffset, copyOutParams, rowTmpGT, sharedBase);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMC2TypeFunc>::Process()
{
    PrepareSendData();
    Communication();
    Calculate();
    if (aivId_ == 0) {
        hccl_.Finalize();
    }
}

}  // namespace MoeDistributeCombineA5Impl
#endif  // MOE_DISTRIBUTE_COMBINE_IMPL_H