/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_train.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_TRAIN_H_
#define FLASH_ATTENTION_SCORE_TRAIN_H_
#include "flash_attention_score_s1s2_const.h"
#include "dropmask.h"
template <typename INPUT_T, typename T = INPUT_T, ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION,
          LayOutTypeEnum layout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasAtten = false, bool hasDrop = false,
          bool hasRope = false, typename OUTPUT_T = INPUT_T, bool isInfer = false,
          bool isPa = false, bool isFd = false>
class FlashAttentionScoreTrain
    : public FlashAttentionScoreS1s2Const<FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>, CHILD_SPEC_TEMPLATE_ARGS> {
public:
    using BaseClass = FlashAttentionScoreS1s2Const<FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>, CHILD_SPEC_TEMPLATE_ARGS>;
    /* =====================GM变量==================== */
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    /* =====================UB变量==================== */
    TBuf<> dropMaskBuf;
    TBuf<> dropMaskIndexBuf;
    TQue<QuePosition::VECIN, 1> dropMaskInQue;

    DropMaskInfo dropMaskInfo;
    __aicore__ inline void InitUniqueOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut) {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
    }
    
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
    __aicore__ inline void Process();
    __aicore__ inline GlobalTensor<INPUT_T> GetKeyGm(RunInfo<isInfer> &runInfo);
    __aicore__ inline GlobalTensor<INPUT_T> GetValueGm(RunInfo<isInfer> &runInfo);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, LocalTensor<uint8_t> &dropMaskUb);
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);

private:
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void CalS1OuterSize(const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetS2LoopRange(RunParamStr<isInfer> &runParam);
};

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueInput(
    __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * dropMask, __gm__ uint8_t * softmaxMax,
    __gm__ uint8_t * softmaxSum, __gm__ uint8_t * actualSeqLengths, __gm__ uint8_t * actualSeqLengthsKv,
    __gm__ uint8_t * blockTable, __gm__ uint8_t * queryPaddingSize, __gm__ uint8_t * kvPaddingSize, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t * &workspace)
{
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    if (hasDrop) {
        this->dropMaskInfo.boolMode = this->tilingData->inputParamsRegbase.needDropMaskOp == 1;
        if (this->dropMaskInfo.boolMode) {
            this->dropMaskGm.SetGlobalBuffer(workspace);
            workspace += this->tilingData->dropmaskParamsRegbase.shapeTotalSize;
        } else {
            this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
        }
    }

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        this->actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        this->actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueConstInfo(
    const InputParamsRegbase &inputParamsRegbase)
{
    this->constInfo.gS1o = this->constInfo.gSize * this->constInfo.s1OuterSize;
    this->constInfo.n2GS1o = this->constInfo.n2Size * this->constInfo.gS1o;
    if constexpr (hasDrop == true) {
        this->dropMaskInfo.seed = inputParamsRegbase.seed;
        this->dropMaskInfo.offset = inputParamsRegbase.offset;
        this->constInfo.keepProb = inputParamsRegbase.keepProb;
        this->dropMaskInfo.keepProbUint8 = static_cast<uint8_t>(inputParamsRegbase.keepProbUint8);
        this->dropMaskInfo.dropMaskOuter = inputParamsRegbase.dropMaskOuter;
    }
};

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueLocalBuffer()
{
    if constexpr (hasDrop) {
        if constexpr (!IsSameType<INPUT_T, float>::value || !BaseClass::containAllOptionalInput) {
            this->pipe->InitBuffer(this->dropMaskInQue, 1, 8192);
        }
        this->pipe->InitBuffer(this->dropMaskIndexBuf, 2048);
        this->pipe->InitBuffer(this->dropMaskBuf, 1024);
    }
};

CHILD_SPEC_TEMPLATE
__aicore__ inline void
FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    // 训练的TND场景的Mask以及Pse都是不带padding的
    runInfo.b1SSOffset = runParam.b1SSOffset;
    // 训练的非TND场景的mask的sequence length一定等于qk的sequence length
    runInfo.b1SSAttenMaskOffset = runParam.b1SSOffset;
    if constexpr (hasDrop) {
        runInfo.b1SSOffsetAlign = runParam.b1SSOffsetAlign16;
    }

    // 训练的preTokens和NextTokens已经被转换好
    if constexpr (hasAtten) {
        runInfo.preTokensPerBatch = this->attenMaskInfo.preTokens;
        runInfo.nextTokensPerBatch = this->attenMaskInfo.nextTokens;
    }

    runInfo.vecCoreOffset = this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = (this->blockIdx >> 1) * this->tilingData->multiCoreParamsRegbase.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParamsRegbase.splitFactorSize;
    if (this->tilingData->multiCoreParamsRegbase.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParamsRegbase.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    RunParamStr<isInfer> runParam;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        CalS1OuterSize(multiCoreInnerOffset, runParam);
    }

    RunInfo<isInfer> runInfo[4];
    int64_t taskId = 0;
    bool notThirdLast = true;
    bool notSecondLast = true;
    bool notLast = true;
    int64_t thirdLast = multiCoreInnerLimit;
    int64_t secondLast = multiCoreInnerLimit + 1;
    int64_t last = multiCoreInnerLimit + 2;
    multiCoreInnerLimit += 3;
    LocalTensor<INPUT_T> scmTensor[2];
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         multiCoreInnerIdx++) {
        if (multiCoreInnerIdx == secondLast) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == last) {
            notLast = false;
        } else if (multiCoreInnerIdx == thirdLast) {
            notThirdLast = false;
        }

        int64_t s2LoopLimit = 0;
        runParam.s2LoopStartIdx = 0;
        bool notLastThreeLoop = notThirdLast && notSecondLast && notLast;
        bool notLastTwoLoop = notSecondLast && notLast;
        if (notLastThreeLoop) {
            this->ComputeAxisIdx(multiCoreInnerIdx, runParam);
            this->GetQueryOffset(runParam);
            // s2轴循环计数, 支持sparse和非sparse场景
            this->GetS2LoopRange(runParam);
            s2LoopLimit = CeilDiv(runParam.s2LineEndIdx - runParam.s2LineStartIdx, this->s2BaseSize) - 1;
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
                } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                    flag.ropeKa = this->constInfo.bN2GDR;
                    flag.ropeKb = this->constInfo.bN2DR;
                } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                    flag.ropeKa = this->constInfo.dSizeRope;
                    flag.ropeKb = this->constInfo.dSizeRope;
                }
            }
            flag.dSize = this->constInfo.dSize;
            flag.dSizeRope = this->constInfo.dSizeRope;
        }
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION ||
                      IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                this->softMaxCheckRes = true;
            }
        }
        bool isLastBmm1 = (multiCoreInnerIdx == multiCoreInnerLimit - 4);
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            RunInfo<isInfer> &info0 = runInfo[taskId & 3];
            if (taskId >= 1 && notLastTwoLoop) {
                this->WaitBmm1Result();
            }

            if (notLastThreeLoop) {
                this->SetRunInfo(info0, runParam, taskId, s2LoopCount, s2LoopLimit,
                                 multiCoreInnerIdx);

                if constexpr (hasRope) {
                    this->IterateBmm1WithRope(info0, runParam, flag,
                                              isLastBmm1 && (s2LoopCount == s2LoopLimit));
                } else {
                    this->IterateBmm1(info0, runParam, flag,
                                      isLastBmm1 && (s2LoopCount == s2LoopLimit));
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
            taskId++;
        }
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline GlobalTensor<INPUT_T> FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::GetKeyGm(
    RunInfo<isInfer> & runInfo)
{
    return this->keyGm;
}

CHILD_SPEC_TEMPLATE
__aicore__ inline GlobalTensor<INPUT_T> FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::GetValueGm(
    RunInfo<isInfer> & runInfo)
{
    return this->valueGm;
}
CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::GenerateDropoutMask(
    RunInfo<isInfer> & runInfo, LocalTensor<uint8_t> & dropMaskUb)
{
    if constexpr (hasDrop == true) {
        if (dropMaskInfo.dropMaskOuter == 1) {
            if constexpr (!IsSameType<INPUT_T, float>::value || !BaseClass::containAllOptionalInput) {
                CopyInDropOuter<hasDrop>(dropMaskBuf, dropMaskInQue, dropMaskGm, runInfo, this->constInfo,
                                         dropMaskInfo);
            } else {
                CopyInDropOuter<hasDrop>(dropMaskBuf, this->attenMaskInQue[1 - runInfo.taskIdMod2], dropMaskGm,
                                         runInfo, this->constInfo, dropMaskInfo);
            }
        } else {
            GenDropMask<hasDrop>(dropMaskBuf, dropMaskIndexBuf, runInfo, this->constInfo, dropMaskInfo);
        }
        dropMaskUb = this->dropMaskBuf.template Get<uint8_t>();
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> & runInfo, LocalTensor<float> & sumUb, LocalTensor<float> & maxUb)
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
    int64_t s1Offset =
        (runInfo.s1oIdx * this->s1BaseSize + this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t gmOffset = (bOffset + n2Offset + gOffset + s1Offset) * fp32BaseSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;

    this->BroadCastAndCopyOut(runInfo, softmaxSumGm, softmaxMaxGm, gmOffset, calculateSize);
}

CHILD_SPEC_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> & runInfo, LocalTensor<VEC2_RES_T> & vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

// =========================================== private functions ===========================================
CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::GetS1LoopRange(
    int64_t & multiCoreInnerOffset, int64_t & multiCoreInnerLimit)
{
    if constexpr (layout != LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!hasAtten) {
            return;
        }
        if (this->tilingData->inputParamsRegbase.sparseType <= 0) {
            return;
        }
    }

    // TND/Sparse 场景下负载均衡后每个核获取的结果
    int32_t aicIdx = this->blockIdx >> 1;
    multiCoreInnerOffset = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    if (likely((this->tilingData->multiCoreParamsRegbase.coreNum - 1) > aicIdx)) {
        multiCoreInnerLimit = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    } else {
        multiCoreInnerLimit = this->tilingData->multiCoreParamsRegbase.totalSize;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::CalS1OuterSize(
    const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam)
{
    int64_t actualS1Outersize = 0;
    runParam.boIdx = 0;
    this->s1OuterSizeAcc = 0;
    runParam.b1SSOffset = 0;
    runParam.b1SSOffsetAlign16 = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;

    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->inputParamsRegbase.bSize; ++i) {
        this->GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (CeilDiv(actualS1Len, this->s1BaseSize) * this->constInfo.n2G);
        if (multiCoreInnerOffset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            runParam.b1SSOffset += actualS1Len * actualS2Len;
            runParam.b1SSOffsetAlign16 += actualS1Len * Align(actualS2Len);
            runParam.boIdx++;
            continue;
        }
        if (actualS2Len == 0 && actualS1Len != 0) {
            int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)this->actualSeqQlenAddr)[i - 1];
            AscendC::InitOutput<OUTPUT_T>(this->attentionOutGm[accumSize * this->constInfo.n2GDv],
                                          actualS1Len * this->constInfo.n2GDv, static_cast<OUTPUT_T>(0.0));
            AscendC::InitOutput<float>(softmaxMaxGm[accumSize * this->constInfo.n2G * 8],
                                       actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
            AscendC::InitOutput<float>(softmaxSumGm[accumSize * this->constInfo.n2G * 8],
                                       actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
        }
        break;
    }
}

CHILD_SPEC_TEMPLATE
__aicore__ inline void FlashAttentionScoreTrain<CHILD_SPEC_TEMPLATE_ARGS>::GetS2LoopRange(RunParamStr<isInfer> &
                                                                                          runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        int64_t actualS1Len;
        int64_t actualS2Len;
        this->GetSeqQlenKvlenByBoidx(runParam.boIdx, actualS1Len, actualS2Len);
        if constexpr (hasAtten) {
            if (this->tilingData->inputParamsRegbase.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, actualS2Len);
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND)) {
                runParam.s2LineStartIdx = Max(
                    runParam.s1oIdx * this->s1BaseSize - this->tilingData->inputParamsRegbase.s1SparseValidSize, 0);
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize +
                                                this->tilingData->inputParamsRegbase.s2SparseValidSize,
                                            actualS2Len);
                // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
                runParam.s2LineStartIdx =
                    Max(runParam.s1oIdx * this->s1BaseSize - actualS1Len +
                            Max(actualS2Len - this->tilingData->inputParamsRegbase.preTokens, 0),
                        0);
                runParam.s2LineEndIdx =
                    Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                            Max(actualS1Len - this->tilingData->inputParamsRegbase.nextTokens, 0),
                        actualS2Len);
                // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
                if (runParam.boIdx == this->tilingData->inputParamsRegbase.bandIndex) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len +
                                                    this->tilingData->inputParamsRegbase.nextTokens - actualS1Len,
                                                actualS2Len);
                } else {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx =
                        Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
                }
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
                if (runParam.boIdx == this->tilingData->inputParamsRegbase.bandIndex) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx =
                        Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                Max(actualS1Len - this->tilingData->inputParamsRegbase.nextTokens, 0),
                            actualS2Len);
                } else {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, actualS2Len);
                }
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Max((runParam.s1oIdx + 1) * this->s1BaseSize - actualS1Len + actualS2Len,
                                            ((__gm__ int64_t *)this->prefixNAddr)[runParam.boIdx]);
                runParam.s2LineEndIdx = Min(runParam.s2LineEndIdx, actualS2Len);
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = actualS2Len;
            }
        } else {
            runParam.s2LineStartIdx = 0;
            runParam.s2LineEndIdx = actualS2Len;
        }

        if constexpr (hasDrop) {
            // 外部传入的drop_mask未进行bit转bool, 需要进行对齐
            runParam.s2LineStartIdx = runParam.s2LineStartIdx >> 4 << 4;
            runParam.s2LineEndIdx = Min(CeilDiv(runParam.s2LineEndIdx, 16) << 4, actualS2Len);
        }
    } else {
        if constexpr (hasAtten) {
            // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
            if (this->tilingData->inputParamsRegbase.sparseType ==
                static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, this->constInfo.s2Size);
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND)) {
                // 对角线往外扩散场景, s1和s2可能不同
                runParam.s2LineStartIdx = Max(
                    runParam.s1oIdx * this->s1BaseSize - this->tilingData->inputParamsRegbase.s1SparseValidSize, 0);
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize +
                                                this->tilingData->inputParamsRegbase.s2SparseValidSize,
                                            this->constInfo.s2Size);
            } else if (this->tilingData->inputParamsRegbase.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx =
                    Max(this->s1BaseSize * (runParam.s1oIdx + 1) - this->constInfo.s1Size + this->constInfo.s2Size,
                        ((__gm__ int64_t *)this->prefixNAddr)[runParam.boIdx]);
                if (runParam.s2LineEndIdx <= runParam.s2LineStartIdx) {
                    // 无效行场景至少要算一个基本块
                    runParam.s2LineEndIdx = this->s2BaseSize;
                } else {
                    runParam.s2LineEndIdx = CeilDiv(runParam.s2LineEndIdx, this->s2BaseSize) * this->s2BaseSize;
                }
                runParam.s2LineEndIdx = Min(runParam.s2LineEndIdx, this->constInfo.s2Size);
            } else { // 其它场景, 如无attention mask
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = this->constInfo.s2Size;
            }
        } else {
            runParam.s2LineStartIdx = 0;
            runParam.s2LineEndIdx = this->constInfo.s2Size;
        }
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode ==
            static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            if (this->tilingData->inputParamsRegbase.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
                // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min(this->constInfo.s2Size, 128L);
                }
            }
        }
    }
    if constexpr (BaseClass::isFp8) {
        runParam.s2LineStartIdx = runParam.s2LineStartIdx >> 7 << 7;
    }

    return;
}
#endif // FLASH_ATTENTION_SCORE_TRAIN_H_