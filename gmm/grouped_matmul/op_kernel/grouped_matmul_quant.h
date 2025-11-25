/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_quant.h
 * \brief 全量化场景Kernel文件，适用耦合架构
 */
#ifndef ASCENDC_GROUPED_MATMUL_QUANT_H
#define ASCENDC_GROUPED_MATMUL_QUANT_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"

#if defined(GMM_QUANT_FLOAT16)
namespace GROUPED_MATMUL {
/*@brief store variables for core split configuration
*/
constexpr int32_t PIPELINE_NUM = 4;
constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint32_t FP32_PER_REPEAT = 64;
constexpr uint32_t FP16_PER_REPEAT = 128;
constexpr uint32_t FP16_BLOCK_VAL_NUM = 16;
constexpr uint32_t FP32_BLOCK_VAL_NUM = 8;
constexpr uint32_t BLOCK_BYTE = 32;
constexpr uint32_t CUBE_SIZE = 256;
constexpr uint32_t REPEAT_BLOCK_NUM = 8;


/** @brief intenal computation class
*/
template <class mmType, bool sync = false>
class GMMQuantCompute : public GMMCompute<mmType, sync> {
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = DTYPE_WEIGHT;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GMMQuantCompute(typename mmType::MT& mm_) : GMMCompute<mmType, sync>(mm_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR group_list,
                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                const GMMBaseParams* __restrict gmmBaseParams,
                                const TCubeTiling* __restrict mmTilingData, TPipe* tPipe);

    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx);

    __aicore__ inline void VectorCompute(MNConfig& mnConfig);

    __aicore__ inline void PostCompute();

 private:
    __aicore__ inline void Dequant(MNConfig& mnConfig);
    __aicore__ inline void DequantCompute(LocalTensor<CT> &mmOutInUb, uint32_t curVecBaseM, uint32_t curVecBaseN);

    __aicore__ inline void SetPerTokenQuantStaticBuffer(const GMMBaseParams* __restrict gmmBaseParams,
                                                        const TCubeTiling* __restrict mmTilingData, GM_ADDR workspace);

    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);

    __aicore__ inline void DataCopyPerToken(MNConfig& mnConfig, uint32_t curBaseM, uint64_t offsetM);

    __aicore__ inline void SetPerTokenQuantRefreshedBuffer(const MNConfig mnConfig);

    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig,
        LocalTensor<CT> &mmOutInUb, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, 
                                                     uint32_t offsetM);

    __aicore__ inline void PerTokenQuant(MNConfig& mnConfig, LocalTensor<CT> &mmOutInUb, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM);

    __aicore__ inline void DataCopyOut(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t curVecBaseN,
                                       uint32_t alignBaseN, uint64_t outOffset);

    __aicore__ inline void VectorTilingCalc(MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, 
                                            uint32_t& vecBaseN, uint32_t& vecBaseM);

    GM_ADDR scaleTensorPtr;
    GM_ADDR perTokenScaleTensorPtr;
    GlobalTensor<DTYPE_SCALE> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<half> pertokenBrcbLocal;
    LocalTensor<float> actResultLocal;
    bool sequentialWrite = true;
    bool isPerTokenQuant;
    uint32_t cubeNum;  // Matmul completions on the kernel
    uint32_t ubBaseM_;
    uint32_t ubBaseN_;
};

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                  GM_ADDR scale, GM_ADDR offset, GM_ADDR antiquantScale,
                                                                  GM_ADDR antiquantOffset, GM_ADDR groupList,
                                                                  GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                                                  const GMMBaseParams* __restrict gmmBaseParams,
                                                                  const TCubeTiling* __restrict mmTilingData,
                                                                  TPipe* tPipe) {
    this->GMMCompute<mmType, sync>::Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList,
        perTokenScale, y, workspace, gmmBaseParams, mmTilingData, tPipe);
    isPerTokenQuant = gmmBaseParams->quantParam == 1;
    scaleTensorPtr = scale;
    perTokenScaleTensorPtr = perTokenScale;
    cubeNum = 0;
    ubBaseM_ = gmmBaseParams->ubBaseK;
    ubBaseN_ = gmmBaseParams->ubBaseN;
    SetPerTokenQuantStaticBuffer(gmmBaseParams, mmTilingData, workspace);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::PostCompute() {

}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig,
                                                                       uint32_t coreIdx) {
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    // init global buffer
    if (this->singleX == 0) {
        this->xGm.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, this->xTensorPtr));
    } else {
        this->xGm.SetGlobalBuffer(GetTensorAddr<AT>(0, this->xTensorPtr) + mnConfig.xBaseOffset);
    }
    GlobalTensor<BT> weightGm = this->SetGlobalBufferW(groupIdx, tailN, mnConfig);

    this->mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    this->mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    this->mm.SetTensorA(this->xGm[xOffset], transposeX);
    this->mm.SetTensorB(weightGm, transposeW);
    this->SetGlobalBufferBias(groupIdx, tailN, mnConfig);
    LocalTensor<CT> mmOutLocal = vecInQueue.AllocTensor<CT>();

    this->mm.Iterate();
    this->mm.GetTensorC(mmOutLocal, 0, sequentialWrite);

    vecInQueue.EnQue(mmOutLocal);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::VectorCompute(MNConfig& mnConfig) {
    SetPerTokenQuantRefreshedBuffer(mnConfig);
    Dequant(mnConfig);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::DequantCompute(LocalTensor<CT> &mmOutInUb, uint32_t curVecBaseM, uint32_t curVecBaseN) {
    LocalTensor<float> mmOutLocalFp32 = mmOutInUb.template ReinterpretCast<float>();
    Cast(mmOutLocalFp32, mmOutInUb, RoundMode::CAST_NONE, curVecBaseM * curVecBaseN);
    uint32_t loopTime = curVecBaseN / FP32_PER_REPEAT;
    uint32_t tailLen = curVecBaseN - loopTime * FP32_PER_REPEAT;
    uint32_t repeatTimes = curVecBaseM;

    BinaryRepeatParams repeatParams;

    repeatParams.dstRepStride = curVecBaseN / FP32_BLOCK_VAL_NUM;
    repeatParams.src0RepStride = curVecBaseN / FP32_BLOCK_VAL_NUM;
    repeatParams.src1RepStride = 0;
    LocalTensor<DTYPE_SCALE> scaleInUb = scaleInQueue.DeQue<DTYPE_SCALE>();
    for (uint32_t i = 0; i < loopTime; ++i) {
        uint32_t offset = i * FP32_PER_REPEAT;
        Mul(mmOutLocalFp32[offset], mmOutLocalFp32[offset], scaleInUb[offset], FP32_PER_REPEAT, repeatTimes, repeatParams);
    }
    uint32_t offset = curVecBaseN - tailLen;
    if (tailLen != 0) {
        Mul(mmOutLocalFp32[offset], mmOutLocalFp32[offset], scaleInUb[offset], tailLen, repeatTimes, repeatParams);
    }
    PipeBarrier<PIPE_V>();
    scaleInQueue.EnQue(scaleInUb);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::ComputeDequantAndActivate(MNConfig& mnConfig,
    LocalTensor<CT> &mmOutInUb,
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM) {

    DequantCompute(mmOutInUb, curVecBaseM, alignBaseN);
    if (isPerTokenQuant) {
        PerTokenQuant(mnConfig, mmOutInUb, curVecBaseM, alignBaseN, offsetM);
    } else {
        LocalTensor<DTYPE_Y> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_Y>();
        LocalTensor<float> mmOutLocalFp32 = mmOutInUb.template ReinterpretCast<float>();
        Cast(yLocalInUb, mmOutLocalFp32, RoundMode::CAST_NONE, curVecBaseM * alignBaseN);
        PipeBarrier<PIPE_V>();
        vecOutQueue.EnQue(yLocalInUb);
    }

    return;
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::PerTokenQuant(MNConfig& mnConfig, LocalTensor<CT> &mmOutInUb, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint32_t alignBaseM = AlignUp(curBaseM, static_cast<uint32_t>(UB_BLOCK_UNIT_SIZE / sizeof(float)));
    LocalTensor<DTYPE_Y> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_Y>();
    // copyIn
    DataCopyPerToken(mnConfig, curBaseM, offsetM);

    LocalTensor<float> scaleInUb = scaleInQueue.DeQue<float>();
    LocalTensor<float> perTokenScaleLocal = scaleInUb[ubBaseN_];

    // brac
    LocalTensor<half> perTokenScaleLocalFp16 = perTokenScaleLocal.template ReinterpretCast<half>();
    Cast(perTokenScaleLocalFp16, perTokenScaleLocal, RoundMode::CAST_NONE, alignBaseM);
    Duplicate(pertokenBrcbLocal, (half)0, alignBaseM * BLOCK_BYTE);
    uint8_t repeatTime = alignBaseM / FP16_BLOCK_VAL_NUM;
    uint8_t tailRepeatTime = alignBaseM - repeatTime * FP16_BLOCK_VAL_NUM;

    if (repeatTime == 0) {
        repeatTime = tailRepeatTime;
        tailRepeatTime = 0;
    }

    uint8_t repeatStride = static_cast<uint8_t>(REPEAT_BLOCK_NUM) * 2;
    Add(pertokenBrcbLocal, pertokenBrcbLocal, perTokenScaleLocalFp16,
        FP16_PER_REPEAT, repeatTime, {1, 1, 0, repeatStride, repeatStride, 1});
    Add(pertokenBrcbLocal[FP16_PER_REPEAT], pertokenBrcbLocal[FP16_PER_REPEAT],
        perTokenScaleLocalFp16, FP16_PER_REPEAT, repeatTime, {1, 1, 0, repeatStride, repeatStride, 1});
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < repeatTime; ++i) {
        Transpose(pertokenBrcbLocal[CUBE_SIZE * i], pertokenBrcbLocal[CUBE_SIZE * i]);
    }
    PipeBarrier<PIPE_V>();

    if (tailRepeatTime != 0) {
        uint32_t srcOffset = repeatTime * repeatStride;
        uint32_t dstOffset = FP16_BLOCK_VAL_NUM * FP16_BLOCK_VAL_NUM * repeatTime;
        Add(pertokenBrcbLocal[dstOffset], pertokenBrcbLocal[dstOffset], perTokenScaleLocalFp16[srcOffset],
            FP16_PER_REPEAT, tailRepeatTime, {1, 1, 0, repeatStride, repeatStride, 1});
        Add(pertokenBrcbLocal[dstOffset + FP16_PER_REPEAT], pertokenBrcbLocal[dstOffset + FP16_PER_REPEAT],
            perTokenScaleLocalFp16[srcOffset], FP16_PER_REPEAT, tailRepeatTime, {1, 1, 0, repeatStride, repeatStride, 1});
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < tailRepeatTime; ++i) {
            Transpose(pertokenBrcbLocal[dstOffset + CUBE_SIZE * i], pertokenBrcbLocal[dstOffset + CUBE_SIZE * i]);
        }
        PipeBarrier<PIPE_V>();
    }

    // mul perToken
    LocalTensor<float> mmOutLocalFp32 = mmOutInUb.template ReinterpretCast<float>();
    LocalTensor<half> mmOutLocalFp16 = mmOutInUb.template ReinterpretCast<half>();
    Cast(mmOutLocalFp16, mmOutLocalFp32, RoundMode::CAST_NONE, curBaseM * alignBaseN);
    repeatStride = alignBaseN * sizeof(half) / UB_BLOCK_UNIT_SIZE;
    uint32_t tailNum = alignBaseN % FP16_PER_REPEAT;
    uint64_t alignedN = alignBaseN - tailNum;
    uint64_t perchannelResOffset = 0;
    while (perchannelResOffset < alignedN) {
        Mul(yLocalInUb[perchannelResOffset], mmOutLocalFp16[perchannelResOffset],
            pertokenBrcbLocal, FP16_PER_REPEAT, curBaseM, {1, 1, 0, repeatStride, repeatStride, 1});
        perchannelResOffset += FP16_PER_REPEAT;
    }
    if (tailNum != 0) {
        Mul(yLocalInUb[alignedN], mmOutLocalFp16[alignedN], pertokenBrcbLocal, tailNum, curBaseM, {1, 1, 0, repeatStride, repeatStride, 1});
    }

    PipeBarrier<PIPE_V>();
    scaleInQueue.EnQue(scaleInUb);
    vecOutQueue.EnQue(yLocalInUb);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::VectorTilingCalc(
    MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, uint32_t& vecBaseN,
    uint32_t& vecBaseM) {
    curCubeSingleN = mnConfig.nIdx == mnConfig.blockDimN - 1 ?
                              mnConfig.n - mnConfig.nIdx * mnConfig.singleN : mnConfig.singleN;
    curCubeSingleM = mnConfig.mIdx == mnConfig.blockDimM - 1 ?
                              mnConfig.m - mnConfig.mIdx * mnConfig.singleM : mnConfig.singleM;
    vecBaseN = Min(ubBaseN_, curCubeSingleN);
    vecBaseM = Min(ubBaseM_, curCubeSingleM);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::Dequant(MNConfig& mnConfig) {
    uint32_t curCubeSingleN;
    uint32_t curCubeSingleM;
    uint32_t vecBaseN;
    uint32_t vecBaseM;
    VectorTilingCalc(mnConfig, curCubeSingleN, curCubeSingleM, vecBaseN, vecBaseM);
    uint32_t curVecBaseN = vecBaseN;
    uint32_t curVecBaseM;
    uint32_t rowLength = sequentialWrite ? curCubeSingleN : mnConfig.n;
    LocalTensor<CT> mmOutInUb = vecInQueue.DeQue<CT>();
    PipeBarrier<PIPE_V>();
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += vecBaseN) {
        if (unlikely(offsetN + vecBaseN >= curCubeSingleN)) { curVecBaseN = curCubeSingleN - offsetN; }
        uint32_t alignBaseN = AlignUp(curVecBaseN, static_cast<uint32_t>(UB_BLOCK_UNIT_SIZE / sizeof(int32_t)));
        uint64_t scaleOffset = mnConfig.nIdx * mnConfig.singleN + offsetN;
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset);
        curVecBaseM = vecBaseM;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) { 
                curVecBaseM = curCubeSingleM - offsetM; 
            }
            // use AscendDequant interface to do perchannel dequant
            ComputeDequantAndActivate(mnConfig, mmOutInUb, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            uint64_t outOffset = (mnConfig.mIdx * mnConfig.singleM + offsetM) * mnConfig.n + \
                                  mnConfig.nIdx * mnConfig.singleN + offsetN;
            DataCopyOut(mnConfig, curVecBaseM, curVecBaseN, alignBaseN, outOffset);
        }
        LocalTensor<DTYPE_SCALE> scaleInUb = scaleInQueue.DeQue<DTYPE_SCALE>();
        scaleInQueue.FreeTensor(scaleInUb);
    }
    vecInQueue.FreeTensor(mmOutInUb);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::DataCopyOut(MNConfig& mnConfig, uint32_t curVecBaseM,
                                                                         uint32_t curVecBaseN, uint32_t alignBaseN,
                                                                         uint64_t outOffset) {
    // Copy the result of vector to yGm.
    LocalTensor<DTYPE_Y> yLocal = vecOutQueue.DeQue<DTYPE_Y>();

    DataCopyParams params;
    params.blockCount = curVecBaseM;
    params.blockLen = curVecBaseN * sizeof(DTYPE_Y) / UB_BLOCK_UNIT_SIZE;
    params.srcStride = static_cast<uint32_t>((alignBaseN - curVecBaseN) * sizeof(DTYPE_Y) / UB_BLOCK_UNIT_SIZE);
    params.dstStride = (mnConfig.n - curVecBaseN) * sizeof(DTYPE_Y) / UB_BLOCK_UNIT_SIZE;
    DataCopy(this->yGm[outOffset], yLocal, params);

    vecOutQueue.FreeTensor(yLocal);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::SetPerTokenQuantStaticBuffer(
    const GMMBaseParams* __restrict gmmBaseParams, const TCubeTiling* __restrict mmTilingData, GM_ADDR workspace) {

    this->pipe->InitBuffer(vecInQueue, 1, mmTilingData->baseN * mmTilingData->baseM * sizeof(CT));

    // ubBaseK as ubBaseM
    this->pipe->InitBuffer(vecOutQueue, 2, gmmBaseParams->ubBaseK * gmmBaseParams->ubBaseN * sizeof(DTYPE_Y));
    uint32_t scaleSize = ubBaseN_ * sizeof(DTYPE_SCALE);
    if (isPerTokenQuant) {
        scaleSize += ubBaseM_ * sizeof(float);
    }
    this->pipe->InitBuffer(scaleInQueue, 2, scaleSize);

    this->pipe->InitBuffer(ubBuf, gmmBaseParams->ubRestBytes);
    LocalTensor<uint8_t> buf = ubBuf.template Get<uint8_t>();
    this->mm.SetLocalWorkspace(buf);

    if (isPerTokenQuant) {
        uint32_t pertokenBrcbSize = gmmBaseParams->ubBaseK * 32;
        uint32_t offsetByte = gmmBaseParams->ubRestBytes - pertokenBrcbSize * 2;
        pertokenBrcbLocal = ubBuf.GetWithOffset<half>(pertokenBrcbSize, offsetByte);
    }

}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::SetPerTokenQuantRefreshedBuffer(const MNConfig mnConfig) {
    // Initialize gm memories that need to be reinitialized due to changes in groupidx.
    // Currently, pertoken quant only supports single-tensor mode, 
    // hence set according to x and weight single-tensor mode.
    // Add an if branch if multi-tensor mode for weght is required.
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE>(0, scaleTensorPtr) + mnConfig.nAxisBaseOffset);
    if (isPerTokenQuant) {
        perTokenScaleGm.SetGlobalBuffer((__gm__ float *)perTokenScaleTensorPtr + mnConfig.mAxisBaseOffset);
    }
    // Add an if branch if multi-tensor mode for y is required.
    this->yGm.SetGlobalBuffer(GetTensorAddr<DTYPE_Y>(0, this->yTensorPtr) + mnConfig.yBaseOffset);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::DataCopyScale(
    uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM copy scale
    DataCopyParams scaleParams;
    scaleParams.blockLen = 1;
    scaleParams.blockCount = curBaseN * sizeof(DTYPE_SCALE) / 32;
    scaleParams.srcStride = 0;
    scaleParams.dstStride = 0;
    LocalTensor<DTYPE_SCALE> scaleLocal = scaleInQueue.AllocTensor<DTYPE_SCALE>();
    DataCopy(scaleLocal, scaleGm[scaleOffset], scaleParams);
    scaleInQueue.EnQue(scaleLocal);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMQuantCompute<mmType, sync>::DataCopyPerToken(
    MNConfig& mnConfig, uint32_t curBaseM, uint64_t offsetM) {
    // GM copy perToken
    LocalTensor<float> scaleInUb = scaleInQueue.DeQue<float>();
    LocalTensor<float> perTokenLocal = scaleInUb[ubBaseN_];
    uint32_t alignBaseM = AlignUp(curBaseM, static_cast<uint32_t>(UB_BLOCK_UNIT_SIZE / sizeof(float)));
    uint64_t perTokenScaleOffset = mnConfig.mIdx * mnConfig.singleM + offsetM;
    DataCopy(perTokenLocal, perTokenScaleGm[perTokenScaleOffset], alignBaseM);
    scaleInQueue.EnQue(scaleInUb);
}


}  // namespace GROUPED_MATMUL

#endif
#endif  // ASCENDC_GROUPED_MATMUL_QUANT_H
