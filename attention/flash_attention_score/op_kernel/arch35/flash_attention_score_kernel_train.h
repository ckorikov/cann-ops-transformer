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
 * \file flash_attention_score_kernel_train.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
#define FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
#include "../../../common/op_kernel/arch35/flash_attention_score_kernel_base.h"
#include "../../../common/op_kernel/arch35/dropmask.h"
namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelTrain
    : public FlashAttentionScoreKernelBase<FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionScoreKernelBase<FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, 
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
private:
    __aicore__ inline void GetAttentionOffset(RunParamStr<isInfer> &runParam);
    __aicore__ inline void CalS1OuterSize(const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetS2LoopRange(RunParamStr<isInfer> &runParam);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    this->constInfo.gS1o = this->constInfo.gSize * this->constInfo.s1OuterSize;
    this->constInfo.n2GS1o = this->constInfo.n2Size * this->constInfo.gS1o;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
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
        if ASCEND_IS_AIV {
            runInfo.preTokensPerBatch = this->attenMaskInfo.preTokens;
            runInfo.nextTokensPerBatch = this->attenMaskInfo.nextTokens;
        }
    }

    runInfo.vecCoreOffset = this->constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->sharedParams.multiCoreInnerOffset;
    int64_t multiCoreInnerLimit = this->sharedParams.multiCoreInnerLimit;
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
            this->GetAttentionOffset(runParam);
            // s2轴循环计数, 支持sparse和非sparse场景
            this->GetS2LoopRange(runParam);
            s2LoopLimit = CeilDiv(runParam.s2LineEndIdx - runParam.s2LineStartIdx, this->s2BaseSize) - 1;
        }
        if ASCEND_IS_AIV {
            if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION ||
                        IsSameType<INPUT_T, float>::value) {
                if (this->sharedParams.implMode ==
                    static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                    this->constInfo.softMaxCheckRes = true;
                }
            }
        }
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (notLastThreeLoop) {
                RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                if ASCEND_IS_AIC {
                    this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                }
            }
            if (taskId > 0 && notLastTwoLoop) {
                if ASCEND_IS_AIV {
                    auto &runInfo3 = runInfo[(taskId + 3) & 3];
                    this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                        this->constInfo);
                }
            }
            if (taskId > 1 && notLast) {
                RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                if ASCEND_IS_AIC {
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    } else {
                        this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    }
                }
            }
            if (taskId > 2) {
                if ASCEND_IS_AIV {
                    RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                    }
                }
            }
            taskId++;
        }
    }
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetAttentionOffset(
    RunParamStr<isInfer> &runParam)
{
    if ASCEND_IS_AIV {
        int64_t bOffsetOut = 0;
        int64_t s1OffsetOut = 0;
        int64_t n2OffsetOut = 0;
        int64_t gOffsetOut = 0;
        int64_t subBlockS1Offset = 0;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffsetOut = this->s1SizeAcc * this->constInfo.n2GDv;
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseN2GDv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gDv;
            gOffsetOut = runParam.goIdx * this->constInfo.dSizeV;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.n2GDv;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffsetOut = runParam.boIdx * this->constInfo.n2GS1Dv;
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseN2GDv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gDv;
            gOffsetOut = runParam.goIdx * this->constInfo.dSizeV;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.n2GDv;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseBN2GDv;
            bOffsetOut = runParam.boIdx * this->constInfo.n2GDv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gDv;
            gOffsetOut = runParam.goIdx * this->constInfo.dSizeV;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.bN2GDv;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffsetOut = runParam.boIdx * this->constInfo.n2GS1Dv;
            n2OffsetOut = runParam.n2oIdx * this->constInfo.gS1Dv;
            gOffsetOut = runParam.goIdx * this->constInfo.s1Dv;
            s1OffsetOut = runParam.s1oIdx * this->constInfo.s1BaseDv;
            subBlockS1Offset = this->constInfo.subBlockIdx * runParam.firstHalfS1RealSize * this->constInfo.dSizeV;
        }
        runParam.attentionOutOffset = bOffsetOut + n2OffsetOut + gOffsetOut + s1OffsetOut + subBlockS1Offset;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::CalS1OuterSize(
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
    for (int64_t i = 0; i < this->sharedParams.bSize; ++i) {
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
        if ASCEND_IS_AIV {
            if (actualS2Len == 0 && actualS1Len != 0) {
                int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)this->actualSeqQlenAddr)[i - 1];
                AscendC::InitOutput<OUTPUT_T>(this->vecBlock.attentionOutGm[accumSize * this->constInfo.n2GDv],
                                              actualS1Len * this->constInfo.n2GDv, static_cast<OUTPUT_T>(0.0));
                AscendC::InitOutput<float>(this->vecBlock.softmaxMaxGm[accumSize * this->constInfo.n2G * 8],
                                           actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
                AscendC::InitOutput<float>(this->vecBlock.softmaxSumGm[accumSize * this->constInfo.n2G * 8],
                                           actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
            }
        }
        break;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetS2LoopRange(
    RunParamStr<isInfer> &runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        int64_t actualS1Len;
        int64_t actualS2Len;
        this->GetSeqQlenKvlenByBoidx(runParam.boIdx, actualS1Len, actualS2Len);
        if constexpr (hasAtten) {
            if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, actualS2Len);
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND)) {
                runParam.s2LineStartIdx = Max(
                    runParam.s1oIdx * this->s1BaseSize - this->sharedParams.s1SparseValidSize, 0);
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize +
                                                this->sharedParams.s2SparseValidSize,
                                            actualS2Len);
                // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
                runParam.s2LineStartIdx =
                    Max(runParam.s1oIdx * this->s1BaseSize - actualS1Len +
                            Max(actualS2Len - this->sharedParams.preTokens, 0),
                        0);
                runParam.s2LineEndIdx =
                    Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                            Max(actualS1Len - this->sharedParams.nextTokens, 0),
                        actualS2Len);
                // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
                if (runParam.s2LineEndIdx - runParam.s2LineStartIdx <= 0) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = actualS2Len;
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
                if (runParam.boIdx == this->sharedParams.bandIndex) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len +
                                                    this->sharedParams.nextTokens - actualS1Len,
                                                actualS2Len);
                } else {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx =
                        Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
                if (runParam.boIdx == this->sharedParams.bandIndex) {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx =
                        Min((runParam.s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                Max(actualS1Len - this->sharedParams.nextTokens, 0),
                            actualS2Len);
                } else {
                    runParam.s2LineStartIdx = 0;
                    runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, actualS2Len);
                }
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Max((runParam.s1oIdx + 1) * this->s1BaseSize - actualS1Len + actualS2Len,
                                            ((__gm__ int64_t *)(this->attenMaskInfo.prefixNAddr))[runParam.boIdx]);
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
            if (this->sharedParams.sparseType ==
                static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize, this->constInfo.s2Size);
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::BAND)) {
                // 对角线往外扩散场景, s1和s2可能不同
                runParam.s2LineStartIdx = Max(
                    runParam.s1oIdx * this->s1BaseSize - this->sharedParams.s1SparseValidSize, 0);
                runParam.s2LineEndIdx = Min((runParam.s1oIdx + 1) * this->s1BaseSize +
                                                this->sharedParams.s2SparseValidSize,
                                            this->constInfo.s2Size);
            } else if (this->sharedParams.sparseType ==
                       static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
                runParam.s2LineStartIdx = 0;
                runParam.s2LineEndIdx =
                    Max(this->s1BaseSize * (runParam.s1oIdx + 1) - this->constInfo.s1Size + this->constInfo.s2Size,
                        ((__gm__ int64_t *)(this->attenMaskInfo.prefixNAddr))[runParam.boIdx]);
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
        if (this->sharedParams.implMode ==
            static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            if (this->sharedParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
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
}
#endif // FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_