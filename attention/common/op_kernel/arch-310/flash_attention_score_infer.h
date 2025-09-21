/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_infer.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_INFER_H_
#define FLASH_ATTENTION_SCORE_INFER_H_
#include "flash_attention_score_s1s2_const.h"
#include "vf/vf_flash_decode.h"
#include "vf/vf_post_quant.h"
#include "infer_flash_attention_comm.h"
#include "infer_flash_attention_kvcache.h"
#include "infer_flash_attention_sparse.h"

template <typename INPUT_T, typename T = INPUT_T, ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION,
          LayOutTypeEnum layout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasAtten = false, bool hasDrop = false, bool hasRope = false,
          typename OUTPUT_T = INPUT_T, bool isInfer = false, bool isPa = false, bool isFd = false>
class FlashAttentionScoreInfer : public FlashAttentionScoreS1s2Const<FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>, CHILD_SPEC_TEMPLATE_ARGS> {
public:
    /* =====================常量==================== */
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using BaseClass = FlashAttentionScoreS1s2Const<FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>, CHILD_SPEC_TEMPLATE_ARGS>;
    /* =====================GM变量==================== */
    __gm__ uint8_t *currentKey;    // pageattention需要
    __gm__ uint8_t *currentValue;  // pageattention需要
    __gm__ uint8_t *blocktablePtr; // pageattention需要

    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> softmaxFDMaxGm;
    GlobalTensor<T> softmaxFDSumGm;

    GlobalTensor<float> postQuantScaleGm;
    GlobalTensor<float> postQuantOffsetGm;
    GlobalTensor<bfloat16_t> postQuantScaleBf16Gm;
    GlobalTensor<bfloat16_t> postQuantOffsetBf16Gm;

    /* =====================UB变量==================== */
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue; // FD
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue; // FD
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;; // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;; // postQuant

    __aicore__ inline void InitUniqueOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut);
    __aicore__ inline void InitUniqueInput(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
                                           __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                           __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                                           __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
                                           __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                           __gm__ uint8_t *&workspace);
    __aicore__ inline void InitUniqueConstInfo(const InputParamsRegbase &inputParamsRegbase);
    __aicore__ inline void InitUniqueLocalBuffer();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, 
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline GlobalTensor<INPUT_T> GetKeyGm(RunInfo<isInfer> &runInfo);
    __aicore__ inline GlobalTensor<INPUT_T> GetValueGm(RunInfo<isInfer> &runInfo);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, LocalTensor<uint8_t> &dropMaskUb)
    {
    }
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);
    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx);

    __aicore__ inline void FDPostQuant(LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount);

    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);

private:
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitLseOutputSingleCore();
    __aicore__ inline void InitFDBuffers();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                    RunParamStr<isInfer> &runParam);
    __aicore__ inline void FlashDecodeCompute();
    __aicore__ inline void GetActualSeqLenKV(int64_t boIdx, int64_t &actualSeqKvLen);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo<isInfer> &runInfo);
    __aicore__ inline void CombineSplitKVRes(uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint32_t splitSize,
                                             uint64_t lseOffset);

    __aicore__ inline void Bmm2FDOut(RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize);

    __aicore__ inline void CopyLseIn(uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void CopyFinalResOut(uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow,
                                           uint32_t dealRowCount, uint64_t perChannelQuantOffset);

    __aicore__ inline void CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo);

    __aicore__ inline void ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                          uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void ReduceFDDataCopyOut(uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb,
                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount);
};

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueOutput(__gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut) 
{
    this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    this->constInfo.isSoftmaxLseEnable = this->tilingData->inputParamsRegbase.isSoftMaxLseEnable;
    if (this->tilingData->initOutputParams.needInit == 1) {
        InitOutputSingleCore();
        // lse output
        if (this->constInfo.isSoftmaxLseEnable) {
            SyncAll();
            InitLseOutputSingleCore();
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueInput(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *&workspace)
{
    if (this->tilingData->inputParamsRegbase.fromFused) {
        ListTensorDesc keyListTensorDescInit((__gm__ void *)key);
        ListTensorDesc valueListTensorDescInit((__gm__ void *)value);
        currentKey = (__gm__ uint8_t *)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        currentValue = (__gm__ uint8_t *)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        if (this->constInfo.isKvContinuous == 1) {
            this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
            this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
        } else {
            this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
            this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
        }
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }
    if constexpr (isPa) {
        blocktablePtr = blockTable;
    }

    if constexpr (isFd) {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        uint64_t accumOutSize = this->tilingData->inputParamsRegbase.accumOutSize;
        uint64_t logSumExpSize = this->tilingData->inputParamsRegbase.logSumExpSize;
        accumOutGm.SetGlobalBuffer((__gm__ T *)(workspace));
        workspace += accumOutSize * sizeof(float);
        softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
        softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
    }
    if constexpr (POST_QUANT) {
        this->InitPostQuant(postQuantScale, postQuantOffset);
    }

    if (this->constInfo.isQHasLeftPadding) {
        this->constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
        if (this->constInfo.queryRightPaddingSize < 0) {
            this->constInfo.queryRightPaddingSize = 0;
        }
    }
    if (this->constInfo.isKVHasLeftPadding) {
        this->constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
        if (this->constInfo.kvRightPaddingSize < 0) {
            this->constInfo.kvRightPaddingSize = 0;
        }
    }

    if (!this->constInfo.isActualLenDimsNull) {
        this->actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
    }
    if (!this->constInfo.isActualLenDimsKVNull) {
        this->actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
{
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantOffsetExist = false;
        if (!this->constInfo.isPostQuantPerChnl && !this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
                this->constInfo.postQuantScaleValue = postQuantScaleGm.GetValue(0);
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
                this->constInfo.postQuantOffsetValue = postQuantOffsetGm.GetValue(0);
            } else {
                this->constInfo.postQuantOffsetValue = 0.0;
            }
        }

        if (postQuantScale != nullptr && !this->constInfo.isPostQuantPerChnl && this->constInfo.isPostQuantBF16) {
            postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
            this->constInfo.postQuantScaleValue = ToFloat(postQuantScaleBf16Gm.GetValue(0));
        }
        if (!this->constInfo.isPostQuantPerChnl && this->constInfo.isPostQuantBF16) {
            if (postQuantOffset != nullptr) {
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
                this->constInfo.postQuantOffsetValue = ToFloat(postQuantOffsetBf16Gm.GetValue(0));
            } else {
                this->constInfo.postQuantOffsetValue = 0.0;
            }
        }

        if (this->constInfo.isPostQuantPerChnl && !this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                this->postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                this->constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
            }
        }

        if (this->constInfo.isPostQuantPerChnl && this->constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                this->constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
            }
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void
FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueConstInfo(const InputParamsRegbase &inputParamsRegbase)
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = inputParamsRegbase.kvSplitPart;
        this->constInfo.sInnerLoopSize = CeilDivision(this->constInfo.s2Size, this->constInfo.splitKVNum);
        this->constInfo.actualCombineLoopSize = CeilDivision(this->constInfo.s2Size, this->constInfo.sInnerLoopSize);
    }

    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
    }
    this->constInfo.isRowInvalid = inputParamsRegbase.isRowInvalid;
    this->constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
    this->constInfo.isGqa = inputParamsRegbase.isGqa;
    this->constInfo.isKvContinuous = inputParamsRegbase.isKvContinuous;
    this->constInfo.actualSeqLenSize = inputParamsRegbase.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(inputParamsRegbase.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(inputParamsRegbase.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(inputParamsRegbase.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(inputParamsRegbase.isKVHasLeftPadding);
    // pageAttention
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }

    // service vector2
    this->constInfo.isBSNDOut = inputParamsRegbase.isBSNDOut;
    if (this->constInfo.isBSNDOut == 1) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    }

    if ((!this->constInfo.isActualLenDimsNull) || (!this->constInfo.isActualLenDimsKVNull) ||
        this->constInfo.isQHasLeftPadding || this->constInfo.isKVHasLeftPadding ||
        (this->constInfo.isKvContinuous == 0)) {
        this->enableKVPrefetch = false;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueLocalBuffer()
{
    if (this->constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
        this->pipe->InitBuffer(softmaxLseQueue, 1, (this->s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    if constexpr (POST_QUANT) {
        this->pipe->InitBuffer(postQuantScaleQue, 1, 2048);
        if (this->constInfo.isPostQuantOffsetExist) {
            this->pipe->InitBuffer(postQuantOffsetQue, 1, 2048);
        }
    }
    if (this->tilingData->initOutputParams.needInit == 1) {
        SyncAll();
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void
FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo, runInfo.s2LoopCount, runInfo);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->tilingData->multiCoreParamsRegbase.coreNum;
    if constexpr (isFd) {
        actualCoreNums = this->tilingData->inputParamsRegbase.bSize * this->constInfo.n2Size *
                         this->constInfo.splitKVNum; // b * n2 * splitkv
    }
    int32_t aicIdx = this->blockIdx >> 1;
    if (aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    if constexpr (!isFd) {
        bnStartIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx];
        gS1StartIdx = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx];
        if (likely((this->tilingData->multiCoreParamsRegbase.coreNum - 1) > aicIdx)) {
            bnEndIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            // GS1合轴：bn表示bn2；GS1不合轴：bn表示bn1
            bnEndIdx = this->tilingData->inputParamsRegbase.bSize * this->constInfo.n2Size *
                this->constInfo.headNumRatio;
        }
    } else {
        gS1StartIdx = 0;
        bnStartIdx = 0;
        bnEndIdx = 1;
        s2LoopLimit = 0;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    LocalTensor<INPUT_T> scmTensor[2];
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;

    if constexpr (isFd) {
        runParam.boIdx = (this->blockIdx >> 1) / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParam.n2oIdx = ((this->blockIdx >> 1) / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    // 注意这里不等于0是因为，推理的在SetRunInfo中第一次也需要赋值runInfo.s1oIdx，boIdx，n2oIdx，goIdx
    // 训练这些值在multiCoreInnerIdx = 0的时候都是0，两边不统一
    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParam.n2oIdx = bnIdx % (this->constInfo.n2Size * this->constInfo.headNumRatio) /
                this->constInfo.headNumRatio;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo, lastBN,
                                                                      nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * ((this->blockIdx >> 1) % this->constInfo.splitKVNum) >
                runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * ((this->blockIdx >> 1) % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1;
        }
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                runParam.s2LoopStartIdx = 0;
                s2LoopLimit = 0;
            }
            FAFlagDataWhole<hasRope> flag = {0};
            if constexpr (hasRope) {
                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                    flag.ropeKa = this->constInfo.n2GDR;
                    flag.ropeKb = this->constInfo.n2DR;
                } else {
                    if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                        flag.ropeKa = this->constInfo.n2GDR;
                        flag.ropeKb = this->constInfo.n2DR;
                    } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                        flag.ropeKa = this->constInfo.dSizeRope;
                        flag.ropeKb = this->constInfo.dSizeRope;
                    }
                }
                flag.dSize = this->constInfo.dSize;
                flag.dSizeRope = this->constInfo.dSizeRope;
            }
            for (int64_t s2LoopCount = runParam.s2LoopStartIdx; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (taskId >= 1 && notLastTwoLoop) {
                    this->WaitBmm1Result();
                }

                if (notLastThreeLoop) {
                    this->SetRunInfo(runInfo[taskId & 3], runParam, taskId, s2LoopCount, s2LoopLimit,
                                     multiCoreInnerIdx);

                    if constexpr (hasRope) {
                        this->IterateBmm1WithRope(runInfo[taskId & 3], runParam, flag, isLastBmm1 && (s2LoopCount == s2LoopLimit));
                    } else {
                        this->IterateBmm1(runInfo[taskId & 3], runParam, flag, isLastBmm1 && (s2LoopCount == s2LoopLimit));
                    }
                }
                if (taskId >= 1 && notLastTwoLoop) {
                    this->ProcessVec1(runInfo[(taskId + 3) & 3], scmTensor[taskId & 1]);
                }

                if (taskId > 1 && notLast) {
                    this->IterateBmm2(runInfo[(taskId + 2) & 3], scmTensor[!(taskId & 1)]);
                }

                if (taskId > 2) {
                    this->WaitBmm2Result();
                    this->ProcessVec2(runInfo[(taskId + 1) & 3]);
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::Process()
{
    ProcessMainLoop();
    if constexpr (isFd) {
        SyncAll();
        InitFDBuffers();
        FlashDecodeCompute();
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline GlobalTensor<INPUT_T>
FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::GetKeyGm(RunInfo<isInfer> &runInfo)
{
    GlobalTensor<INPUT_T> tempKeyGm = this->keyGm;
    IterateAllPreProcess<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runInfo, this->constInfo, this->keyGm, tempKeyGm);
    return tempKeyGm;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline GlobalTensor<INPUT_T>
FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::GetValueGm(RunInfo<isInfer> &runInfo)
{
    GlobalTensor<INPUT_T> tempValueGm = this->valueGm;
    IterateAllPreProcess<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runInfo, this->constInfo, this->valueGm,
                                                                     tempValueGm);
    return tempValueGm;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo,
                                                                                              LocalTensor<float> &sumUb,
                                                                                              LocalTensor<float> &maxUb)
{
    if constexpr (isFd) {
        ComputeLogSumExpAndCopyToGm(runInfo);
        return;
    }
    SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
}

CHILD_SPEC_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    if constexpr (isFd) {
        Bmm2FDOut(runInfo, vec2ResUb, vec2CalcSize);
    } else {
        this->Bmm2DataCopyOut(runInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
    }
}

// =========================================== private functions ===========================================
CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitOutputSingleCore()
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - this->blockIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;

    if constexpr (POST_QUANT) {
        // InitOutput(this->attentionOutGm[this->blockIdx * initParams.singleCoreSize], singleInitOutputSize, (OUTPUT_T)0);
    } else {
        InitOutput<OUTPUT_T>(this->attentionOutGm[this->blockIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitLseOutputSingleCore()
{
    int64_t tmpBlockIdx = this->blockIdx;
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && tmpBlockIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (tmpBlockIdx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[tmpBlockIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)], 
            singleCoreLseSize, 3e+99); // 3e+99:set the value of invalid batch to inf
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::InitFDBuffers()
{
    this->pipe->Reset();
    this->pipe->InitBuffer(lseTmpBuff, bufferSizeByte32K);
    this->pipe->InitBuffer(softmaxMaxInputQue, 1, bufferSizeByte32K);
    this->pipe->InitBuffer(softmaxSumInputQue, 1, bufferSizeByte32K);
    this->pipe->InitBuffer(FDResOutputQue, 1, bufferSizeByte32K);
    this->pipe->InitBuffer(accumOutInputQue, 1, bufferSizeByte32K);
    if constexpr (POST_QUANT) {
        this->pipe->InitBuffer(postQuantScaleQue, 1, bufferSizeByte32K);
        if (this->constInfo.isPostQuantOffsetExist) {
            this->pipe->InitBuffer(postQuantOffsetQue, 1, bufferSizeByte32K);
        }
    }
    if (this->constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND, 每行结果存为8个重复lse元素(32B对齐)
        this->pipe->InitBuffer(softmaxLseQueue, 1, (this->s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    // GS1合轴时，g轴信息包含在gS1中；GS1不合轴时，g轴信息包含在bn2g中；
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::FlashDecodeCompute()
{
    int64_t bIdx = this->blockIdx / this->constInfo.n2Size;
    int64_t n2Idx = this->blockIdx % this->constInfo.n2Size;
    int64_t batchSize = this->tilingData->inputParamsRegbase.bSize;
    if (this->blockIdx >= batchSize * this->constInfo.n2Size) {
        return;
    }
    int64_t actualSeqLen;
    GetActualSeqLenKV(bIdx, actualSeqLen);
    if (actualSeqLen == 0) {
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * this->constInfo.n2GDv + n2Idx * this->constInfo.gDv;
    CombineSplitKVRes(attenOutOffset, bIdx, n2Idx);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::GetActualSeqLenKV(int64_t boIdx,
                                                                                             int64_t &actualSeqLen)
{
    int64_t s2InCurrentBatch = this->constInfo.s2Size;
    if (this->constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)this->keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, boIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    if (this->constInfo.isActualLenDimsKVNull) {
        actualSeqLen = s2InCurrentBatch;
    } else {
        actualSeqLen = (this->constInfo.actualSeqLenKVSize == 1) ? this->actualSeqKvlenAddr[0] :
                                                                   this->actualSeqKvlenAddr[boIdx];
    }
    if (this->constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = this->constInfo.s2Size - actualSeqLen - this->constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLen = 0;
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    if (!this->constInfo.isSoftmaxLseEnable) {
        return;
    }
    LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
    ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
    softmaxLseQueue.template EnQue(lseUb);
    softmaxLseQueue.DeQue<float>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = runInfo.halfS1RealSize;
    intriParams1.srcStride = 0;
    if (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (this->constInfo.isGqa) {
            intriParams1.dstStride = 0;
        } else {
            intriParams1.dstStride = sizeof(float) * (this->constInfo.n2G - 1);
        }
    } else {
        intriParams1.dstStride = 0;
    }
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);
    softmaxLseQueue.FreeTensor(lseUb);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::CombineSplitKVRes(uint64_t attenOutOffset,
                                                                                             uint32_t bIdx,
                                                                                             uint32_t n2Idx)
{
    uint32_t gSplitSizeLse =
        bufferSizeByte32K / (FA_BYTE_BLOCK * this->constInfo.splitKVNum); // 32K / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = bufferSizeByte32K / sizeof(float) / (uint32_t)dVTemplateType;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (this->constInfo.gSize > gSplitMax) {
        gSplitSize = (gSplitSize > gSplitMax) ? gSplitMax : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > this->constInfo.gSize) ? this->constInfo.gSize : gSplitSize;
    }
    uint32_t loopCount = CeilDivision(this->constInfo.gSize, gSplitSize);
    uint32_t tailSplitSize = this->constInfo.gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>(); // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, fp32BaseSize};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape

    uint64_t perChannelQuantOffset = n2Idx * this->constInfo.dSizeV * this->constInfo.gSize;

    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, gSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * this->constInfo.n2Size + n2Idx) * this->constInfo.gSize + i * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, gSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, gSplitSize, perChannelQuantOffset);
    }
    // 尾块处理
    if (tailSplitSize > 0) {
        uint32_t startRow = (loopCount - 1) * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, tailSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * this->constInfo.n2Size + n2Idx) * this->constInfo.gSize + (loopCount - 1) * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, tailSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, tailSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, tailSplitSize, perChannelQuantOffset);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::ComputeScaleValue(LocalTensor<T> lseMaxUb,
                                                                                             LocalTensor<T> lseSumUb,
                                                                                             uint32_t splitSize,
                                                                                             uint64_t lseOffset)
{
    LocalTensor<T> lseOutputUb;
    if (this->constInfo.isSoftmaxLseEnable) {
        lseOutputUb = softmaxLseQueue.template AllocTensor<T>();
    }
    ComputeScaleValue_VF(lseMaxUb, lseSumUb, lseOutputUb, splitSize, this->constInfo.actualCombineLoopSize,
                         this->constInfo.isSoftmaxLseEnable);
    if (this->constInfo.isSoftmaxLseEnable) {
        softmaxLseQueue.template EnQue<T>(lseOutputUb);
        softmaxLseQueue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = splitSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[lseOffset], lseOutputUb, intriParams1);
        softmaxLseQueue.FreeTensor(lseOutputUb);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::Bmm2FDOut(RunInfo<isInfer> &runInfo,
                                                                                     LocalTensor<T> &vec2ResUb,
                                                                                     int64_t vec2CalcSize)
{
    LocalTensor<T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;

    SetFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    WaitFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    attenOut = vec2ResUb;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.halfS1RealSize;
    dataCopyParams.blockLen = this->constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - this->constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    uint32_t mStart = this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t base = (runInfo.boIdx * this->constInfo.n2Size * this->constInfo.gSize * this->constInfo.dSizeV +
                   runInfo.n2oIdx * this->constInfo.gSize * this->constInfo.dSizeV) *
                      this->constInfo.splitKVNum +
                  mStart * this->constInfo.dSizeV;

    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * this->constInfo.gSize * this->constInfo.dSizeV],
                attenOut, dataCopyParams);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::CopyLseIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = this->constInfo.splitKVNum;
    copyInParams.blockLen = dealRowCount * fp32BaseSize * sizeof(T);
    copyInParams.srcStride = (this->constInfo.gSize - dealRowCount) * fp32BaseSize * sizeof(T);
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t combineLseOffset =
        ((uint64_t)bIdx * this->constInfo.n2Size * this->constInfo.splitKVNum + n2Idx * this->constInfo.splitKVNum) *
            this->constInfo.gSize * fp32BaseSize +
        startRow * fp32BaseSize;

    DataCopyPad(softmaxMaxLocal, softmaxFDMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxFDSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::CopyFinalResOut(
    uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t perChannelQuantOffset)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = FDResOutputQue.AllocTensor<OUTPUT_T>(); // 复用内存
    uint32_t dSizeAligned64 = (uint32_t)dVTemplateType;
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dSizeAligned64};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape
    if constexpr (!POST_QUANT) {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dSizeAligned64);
    } else {
        FDPostQuant(tmpBmm2ResCastTensor, accumOutLocal, perChannelQuantOffset + startRow * this->constInfo.dSizeV, dealRowCount);
    }
    FDResOutputQue.EnQue(tmpBmm2ResCastTensor);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeAligned64, this->constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx,
                                                                                          LocalTensor<T> &dst,
                                                                                          LocalTensor<T> &lseLocal,
                                                                                          uint32_t startRow,
                                                                                          uint32_t dealRowCount)
{
    for (uint32_t j = 0; j < this->constInfo.actualCombineLoopSize; ++j) {
        // 第一次，mul结果直接放到dst里
        CopyAccumOutIn(bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_const_VF<T, (uint32_t)dVTemplateType>(dst, lseLocal, accumOutLocal, dealRowCount, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx,
                                                                                          uint32_t splitKVIndex,
                                                                                          uint32_t startRow,
                                                                                          uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = this->constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = ((int64_t)dVTemplateType - this->constInfo.dSizeV) / 8; // 8 for align factor

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = ((int64_t)dVTemplateType - this->constInfo.dSizeV) % 8; // 8 for align factor
    copyInPadParams.paddingValue = 0;

    uint64_t combineAccumOutOffset = ((uint64_t)bIdx * this->constInfo.n2Size * this->constInfo.splitKVNum +
                                      n2Idx * this->constInfo.splitKVNum + splitKVIndex) *
                                         this->constInfo.gSize * this->constInfo.dSizeV +
                                     startRow * this->constInfo.dSizeV;
    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void
FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = this->constInfo.n2G * runInfo.s1SizeAcc;
        n2Offset = runInfo.n2oIdx * this->constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * this->constInfo.n2Size * this->constInfo.gS1;
        n2Offset = runInfo.n2oIdx * this->constInfo.gS1;
        gOffset = runInfo.goIdx * this->constInfo.s1Size;
    }
    int64_t s1Offset = runInfo.s1oIdx * this->s1BaseSize + this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    uint32_t mStart = this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t gmOffset =
        runInfo.boIdx * this->constInfo.n2Size * this->constInfo.splitKVNum * this->constInfo.gSize * fp32BaseSize +
        runInfo.n2oIdx * this->constInfo.splitKVNum * this->constInfo.gSize * fp32BaseSize +
        runInfo.flashDecodeS2Idx * this->constInfo.gSize * fp32BaseSize + mStart * fp32BaseSize;
    // Copy sum to gm
    this->BroadCastAndCopyOut(runInfo, softmaxFDSumGm, softmaxFDMaxGm, gmOffset, calculateSize);
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::ReduceFDDataCopyOut(
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FA_BYTE_BLOCK / sizeof(OUTPUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

CHILD_SPEC_TEMPLATE
template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::PostQuantPerChnl(LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<POSTQUANT_PARAMS_T> copyInPadParams;
    copyInParams.blockCount = gSplitSize;
    copyInParams.blockLen = this->constInfo.dSizeV * sizeof(POSTQUANT_PARAMS_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = ((int64_t)dVTemplateType - this->constInfo.dSizeV) / 8; // 8 for align factor 

    LocalTensor<POSTQUANT_PARAMS_T> postQuantScaleUb = this->postQuantScaleQue.template AllocTensor<POSTQUANT_PARAMS_T>();
    DataCopyPad(postQuantScaleUb, postQuantScaleGm[perChannelQuantOffset], copyInParams, copyInPadParams);

    this->postQuantScaleQue.template EnQue(postQuantScaleUb);
    this->postQuantScaleQue.template DeQue<POSTQUANT_PARAMS_T>();
    if (this->constInfo.isPostQuantOffsetExist) {
        LocalTensor<POSTQUANT_PARAMS_T> postQuantOffsetUb = this->postQuantOffsetQue.template AllocTensor<POSTQUANT_PARAMS_T>();
        DataCopyPad(postQuantOffsetUb, postQuantOffsetGm[perChannelQuantOffset], copyInParams, copyInPadParams);
        this->postQuantOffsetQue.template EnQue(postQuantOffsetUb);
        this->postQuantOffsetQue.template DeQue<POSTQUANT_PARAMS_T>();
        PostQuantPerChnlVF<T, OUTPUT_T, Align64Func((uint16_t)dVTemplateType), POSTQUANT_PARAMS_T>(attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, postQuantOffsetUb, gSplitSize, s1RowCount, this->constInfo.dSizeV);
        this->postQuantOffsetQue.FreeTensor(postQuantOffsetUb);
    } else {
        PostQuantPerChnlVF<T, OUTPUT_T, Align64Func((uint16_t)dVTemplateType), POSTQUANT_PARAMS_T>(attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, gSplitSize, s1RowCount, this->constInfo.dSizeV);

    }
    this->postQuantScaleQue.FreeTensor(postQuantScaleUb);

}

CHILD_SPEC_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::PostQuant(RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx)
{
    uint32_t s1RowCount = this->constInfo.isGqa ? 1U : runInfo.vec2S1RealSize; // s1=1, gS合轴, bn2分核
    uint32_t gRowCount = this->constInfo.isGqa ? runInfo.vec2S1RealSize : 1U;
    if (this->constInfo.isPostQuantPerChnl) {
        uint64_t perChannelQuantGQAOffset = runInfo.n2oIdx * this->constInfo.gDv + vec2S1Idx * this->constInfo.dSizeV + this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize * this->constInfo.dSizeV;
        uint64_t perChannelQuantOffset = this->constInfo.isGqa ? perChannelQuantGQAOffset : runInfo.n2oIdx * this->constInfo.gDv;
        uint32_t gSplitSize = this->constInfo.isPostQuantBF16 ? (2048U / (this->constInfo.dSizeV * sizeof(bfloat16_t))) : (2048U / (this->constInfo.dSizeV * sizeof(float)));
        uint32_t loopCount = (gRowCount + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gRowCount - (loopCount - 1) * gSplitSize;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if (i + 1 == loopCount) {
                gSplitSize = tailSplitSize;
            }
            uint32_t splitOffset = startRow * this->constInfo.dSizeV;
            if (this->constInfo.isPostQuantBF16) {
                PostQuantPerChnl(attenOut, vec2ResUb, perChannelQuantOffset + startRow * this->constInfo.dSizeV, gSplitSize, s1RowCount, splitOffset, postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
            } else {
                PostQuantPerChnl(attenOut, vec2ResUb, perChannelQuantOffset + startRow * this->constInfo.dSizeV, gSplitSize, s1RowCount, splitOffset, postQuantScaleGm, postQuantOffsetGm);
            }
        }
    } else {
        PostQuantPerTensorVF<T, OUTPUT_T, Align64Func((uint16_t)dVTemplateType), true>(attenOut, vec2ResUb, this->constInfo.postQuantScaleValue, this->constInfo.postQuantOffsetValue, runInfo.vec2S1RealSize, this->constInfo.dSizeV);
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreInfer<CHILD_SPEC_TEMPLATE_ARGS>::FDPostQuant(LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount)
{
    if (this->constInfo.isPostQuantPerChnl) {
        if (this->constInfo.isPostQuantBF16) {
            PostQuantPerChnl(attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, postQuantScaleBf16Gm, postQuantOffsetBf16Gm); // q_s = 1
        } else {
            PostQuantPerChnl(attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, postQuantScaleGm, postQuantOffsetGm); // q_s = 1
        }
    } else {
        PostQuantPerTensorVF<T, OUTPUT_T, Align64Func((uint16_t)dVTemplateType), true>(attenOut, accumOutLocal, this->constInfo.postQuantScaleValue, this->constInfo.postQuantOffsetValue, dealRowCount, this->constInfo.dSizeV);
    }
}
#endif