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
 * \file flash_attention_score_kernel_train.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
#define FLASH_ATTENTION_SCORE_KERNEL_TRAIN_H_
#include "flash_attention_score_kernel_base.h"
#include "dropmask.h"
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
    __aicore__ inline void CalS1OuterSize(const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetQueryRopeOffset(RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetS2LoopRange(RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetValueOffset(RunInfo<isInfer> &runInfo);
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
            this->GetQueryOffset(runParam);
            if constexpr (hasRope) {
                this->GetQueryRopeOffset(runParam);
            }
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
        this->prefetchArgs.isLast = (multiCoreInnerIdx == multiCoreInnerLimit - 4);
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (notLastThreeLoop) {
                RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit,
                                 multiCoreInnerIdx);
                if ASCEND_IS_AIC {
                    this->prefetchArgs.isLast = this->prefetchArgs.isLast && (s2LoopCount == s2LoopLimit);
                    if (this->constInfo.enableKVPrefetch) {
                        this->SetPrefetchRightArgs(runInfo1);
                    }
                    this->cubeBlock.IterateBmm1(this->bmm1ResBuf[runInfo1.taskIdMod2].template Get<T>(), runInfo1, this->constInfo, 
                        this->prefetchArgs);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(BaseClass::SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + BaseClass::SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                }
            }
            if (taskId > 0 && notLastTwoLoop) {
                if ASCEND_IS_AIV {
                    auto &runInfo3 = runInfo[(taskId + 3) & 3];
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(BaseClass::SYNC_C1_V1_FLAG[runInfo3.taskIdMod2]); // 等待bmm1完成/等待SYNC_C1_V1_FLAG置位
                    LocalTensor<T> inputTensor = this->bmm1ResBuf[runInfo3.taskIdMod2].template Get<T>();
                    Buffer<BufferType::L1, false> outputBuf = this->l1PBuffers.Get();
                    this->vecBlock.ProcessVec1(outputBuf, inputTensor, runInfo3, this->constInfo);
                }
            }
            if (taskId > 1 && notLast) {
                RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(BaseClass::SYNC_V1_C2_FLAG[runInfo2.taskIdMod3]);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + BaseClass::SYNC_V1_C2_FLAG[runInfo2.taskIdMod3]);
                    if (unlikely(this->constInfo.dSize != this->constInfo.dSizeV)) {
                        GetValueOffset(runInfo2);
                    }
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        this->cubeBlock.IterateBmm2(this->bmm2ResBuf[runInfo2.taskIdMod2].template Get<T>(), this->l1PBuffers, runInfo2, this->constInfo);
                    } else {
                        this->cubeBlock.IterateBmm2(this->bmm2ResGm[runInfo2.taskIdMod3], this->l1PBuffers, runInfo2, this->constInfo);
                    }
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(BaseClass::SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + BaseClass::SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                }
            }
            if (taskId > 2) {
                if ASCEND_IS_AIV {
                    RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(BaseClass::SYNC_C2_V2_FLAG[runInfo3.taskIdMod2]); // 等待bmm2完成/等待SYNC_C2_V2_FLAG置位
                    if constexpr (BaseClass::bmm2Write2Ub) {
                        LocalTensor<T> bmm2Res = this->bmm2ResBuf[runInfo3.taskIdMod2].template Get<T>();
                        this->vecBlock.ProcessVec2(bmm2Res, runInfo3, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec2(this->bmm2ResGm[runInfo3.taskIdMod3], runInfo3, this->constInfo);
                    }
                }
            }
            taskId++;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetValueOffset(
    RunInfo<isInfer> &runInfo)
{
    if constexpr (!isInfer) {
        // 计算gm上的offset
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s2Offset = 0;

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffset = runInfo.s2SizeAcc * this->constInfo.n2Dv;
            s2Offset = runInfo.s2StartIdx * this->constInfo.n2Dv + runInfo.s2LoopCount * this->constInfo.s2BaseN2Dv;
            n2Offset = runInfo.n2oIdx * this->constInfo.dSizeV;
        } else {
            if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                // BSH/BSND
                bOffset = runInfo.boIdx * this->constInfo.n2S2Dv;
                s2Offset = runInfo.s2StartIdx * this->constInfo.n2Dv + runInfo.s2LoopCount * this->constInfo.s2BaseN2Dv;
                n2Offset = runInfo.n2oIdx * this->constInfo.dSizeV;
            } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                // SBH/SBND
                s2Offset = runInfo.s2StartIdx * this->constInfo.mm2Kb + runInfo.s2LoopCount * this->constInfo.s2BaseBN2Dv;
                bOffset = runInfo.boIdx * this->constInfo.n2Dv;
                n2Offset = runInfo.n2oIdx * this->constInfo.dSizeV;
            } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                // BNSD
                bOffset = runInfo.boIdx * this->constInfo.n2S2Dv;
                n2Offset = runInfo.n2oIdx * this->constInfo.s2Dv;
                s2Offset = runInfo.s2StartIdx * this->constInfo.dSizeV + runInfo.s2LoopCount * this->constInfo.s2BaseDv;
            }
        }
        runInfo.valueOffset = bOffset + n2Offset + s2Offset;
    }
}

// =========================================== private functions ===========================================
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
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetQueryRopeOffset(
    RunParamStr<isInfer> &runParam)
{
    // 计算gm上的offset
    int64_t bOffsetRope = 0;
    // s1需要考虑inner轴的影响
    int64_t s1OffsetRope = 0;
    int64_t n2OffsetRope = 0;
    int64_t gOffsetRope = 0;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffsetRope = this->s1SizeAcc * this->constInfo.n2GDR;
        s1OffsetRope = runParam.s1oIdx * this->constInfo.s1BaseN2GDR;
        n2OffsetRope = runParam.n2oIdx * this->constInfo.gDR;
        gOffsetRope = runParam.goIdx * this->constInfo.dSizeRope;
    } else {
        if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffsetRope = runParam.boIdx * this->constInfo.n2GS1DR;
            s1OffsetRope = runParam.s1oIdx * this->constInfo.s1BaseN2GDR;
            n2OffsetRope = runParam.n2oIdx * this->constInfo.gDR;
            gOffsetRope = runParam.goIdx * this->constInfo.dSizeRope;
        } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            s1OffsetRope = runParam.s1oIdx * this->constInfo.s1BaseBN2GDR;
            bOffsetRope = runParam.boIdx * this->constInfo.n2GDR;
            n2OffsetRope = runParam.n2oIdx * this->constInfo.gDR;
            gOffsetRope = runParam.goIdx * this->constInfo.dSizeRope;
        } else if (this->constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffsetRope = runParam.boIdx * this->constInfo.n2GS1DR;
            n2OffsetRope = runParam.n2oIdx * this->constInfo.gS1DR;
            gOffsetRope = runParam.goIdx * this->constInfo.s1DR;
            s1OffsetRope = runParam.s1oIdx * this->constInfo.s1BaseDR;
        }
    }
    runParam.qRopeNBGOffset = bOffsetRope + n2OffsetRope + gOffsetRope + s1OffsetRope;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType>::GetS2LoopRange(RunParamStr<isInfer> &
                                                                                          runParam)
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