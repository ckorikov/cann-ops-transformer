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
 * \file flash_attention_score_s1s2_const.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_S1S2_CONST_H_
#define FLASH_ATTENTION_SCORE_S1S2_CONST_H_
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "matmul_modules/fa_policy_selector.h"
#include "attenmask.h"
#include "pse.h"
#include "infer_flash_attention_comm.h"
#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

static constexpr uint32_t FA_BYTE_BLOCK = 32;

__aicore__ constexpr MatmulConfig GetMm1Cfg(
    bool isFp32, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
    PseTypeEnum pseMode, bool hasAtten, bool hasDrop)
{
    if (isFp32) {
        if ((uint16_t)dTemplateType > (uint16_t)DTemplateType::Aligned256) {
            return GetFACustomCfg(
                true, IterateMode::ITERATE_MODE_ALL, false/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType >> SHIFT_NUM_2,
                false);
        } else if ((uint16_t)dTemplateType <= (uint16_t)DTemplateType::Aligned128) {
            return GetFACustomCfg(
                true, IterateMode::ITERATE_MODE_ALL, true/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                true);
        }
        return GetFACustomCfg(
                true, IterateMode::ITERATE_MODE_ALL, false/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType >> 1,
                true);
    }
    if (dTemplateType == DTemplateType::Aligned64 && s2TemplateType == S2TemplateType::Aligned256 &&
        s1TemplateType == S1TemplateType::Aligned128 && pseMode == PseTypeEnum::PSE_NONE_TYPE &&
        !hasAtten && !hasDrop) {
        // DN高轴reduce场景 SAMEAB场景
        return GetFACustomCfg(
            false, IterateMode::ITERATE_MODE_ALL, true/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
            (uint16_t)s2TemplateType, (uint16_t)s1TemplateType, (uint16_t)dTemplateType,
            (uint16_t)s2TemplateType, (uint16_t)s1TemplateType, (uint16_t)dTemplateType, true);
    }
    if ((uint16_t)dTemplateType > (uint16_t)DTemplateType::Aligned128) {
        if ((uint16_t)dTemplateType > (uint16_t)DTemplateType::Aligned256) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType >> 1,
                false);
        }
        if (pseMode == PseTypeEnum::PSE_NONE_TYPE && !hasAtten && !hasDrop &&
            s1TemplateType != S1TemplateType::Aligned64) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s2TemplateType, (uint16_t)s1TemplateType, (uint16_t)dTemplateType,
                (uint16_t)s2TemplateType, (uint16_t)s1TemplateType, (uint16_t)dTemplateType,
                true);
        } else {
            if (s2TemplateType == S2TemplateType::Aligned256) {
                return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType >> 1, (uint16_t)dTemplateType,
                    true);
            } else {
                return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    true);
            }
        }
    } else {
        if (pseMode == PseTypeEnum::PSE_NONE_TYPE && !hasAtten && !hasDrop &&
            s1TemplateType != S1TemplateType::Aligned64) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_64K/*sharedC1BufferSize*/,
                (uint16_t)s2TemplateType, (uint16_t)s1TemplateType, (uint16_t)dTemplateType,
                (uint16_t)s2TemplateType, (uint16_t)s1TemplateType, (uint16_t)dTemplateType,
                true, true);
        } else {
            if (s2TemplateType == S2TemplateType::Aligned256) {
                return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_64K/*sharedC1BufferSize*/,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    true);
            } else {
                return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_64K/*sharedC1BufferSize*/,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    (uint16_t)s1TemplateType, (uint16_t)s2TemplateType, (uint16_t)dTemplateType,
                    true, true);
            }
        }
    }
}

__aicore__ constexpr uint16_t Align64Func(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

__aicore__ constexpr MatmulConfig GetMm2Cfg(
    bool isFp32, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dQTemplateType,
    DTemplateType dVTemplateType, PseTypeEnum pseMode, bool hasAtten, bool hasDrop)
{
    if (isFp32) {
        if ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType), (uint16_t)s2TemplateType,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType) >> SHIFT_NUM_2, (uint16_t)s2TemplateType,
                false);
        } else if ((uint16_t)dVTemplateType <= (uint16_t)DTemplateType::Aligned128) {
            return GetFACustomCfg(
                true, IterateMode::ITERATE_MODE_ALL, true/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType), (uint16_t)s2TemplateType,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType), (uint16_t)s2TemplateType,
                true);
        }
        return GetFACustomCfg(
                true, IterateMode::ITERATE_MODE_ALL, false/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType), (uint16_t)s2TemplateType,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType), (uint16_t)s2TemplateType >> 1,
                true);
    }
    if (dVTemplateType == DTemplateType::Aligned64 && s2TemplateType == S2TemplateType::Aligned256 &&
        s1TemplateType == S1TemplateType::Aligned128 && pseMode == PseTypeEnum::PSE_NONE_TYPE &&
        !hasAtten && !hasDrop) {
        // DN高轴reduce场景 SAMEB场景
        return GetFACustomCfg(
            false, IterateMode::ITERATE_MODE_ALL, true/*isC1Shared*/, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
            (uint16_t)s1TemplateType, (uint16_t)dVTemplateType, (uint16_t)s2TemplateType,
            (uint16_t)s1TemplateType, (uint16_t)dVTemplateType, (uint16_t)s2TemplateType >> 1, true);
    }
    if ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned128) {
        if ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType), (uint16_t)s2TemplateType,
                (uint16_t)s1TemplateType, Align64Func((uint16_t)dVTemplateType) >> 1, (uint16_t)s2TemplateType,
                false);
        }
        if (s2TemplateType == S2TemplateType::Aligned256) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType >> 1,
                true);
        } else {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_128K/*sharedC1BufferSize*/,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                true);
        }
    } else {
        if (s2TemplateType == S2TemplateType::Aligned256) {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_64K/*sharedC1BufferSize*/,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                true);
        } else {
            return GetFACustomCfg(true, IterateMode::ITERATE_MODE_ALL, true, L0C_SHARED_SIZE_64K/*sharedC1BufferSize*/,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                (uint16_t) s1TemplateType, Align64Func((uint16_t) dVTemplateType), (uint16_t) s2TemplateType,
                true, (uint16_t)dQTemplateType <= (uint16_t)DTemplateType::Aligned128);
        }
    }
}

__aicore__ constexpr bool ContainOptionalInput(
    PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if (pseMode == PseTypeEnum::PSE_NONE_TYPE && !hasAtten && !hasDrop) {
        return false;
    } else {
        return true;
    }
}
template <typename INPUT_T>
__aicore__ constexpr bool IsFp8OnlyWithAttenMask(
    PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if constexpr (!IsSameType<INPUT_T, fp8_e5m2_t>::value &&
                  !IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
                  !IsSameType<INPUT_T, hifloat8_t>::value) {
        return false;
    }
    if (pseMode == PseTypeEnum::PSE_NONE_TYPE && hasAtten && !hasDrop) {
        return true;
    }
    return false;
}

__aicore__ constexpr bool IsDn(
    bool isFp32, PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool isS1Base64,
    DTemplateType dTemplateType, bool hasRope) {
    if (!isFp32 && !ContainOptionalInput(pseMode, hasAtten, hasDrop) && !isS1Base64 &&
        (uint16_t)dTemplateType <= (uint16_t)DTemplateType::Aligned256 && !hasRope) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool UbOutCondition(
    bool isFp32, PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool isS2Base64) {
    if (IsFp8OnlyWithAttenMask<INPUT_T>(pseMode, hasAtten, hasDrop)) {
        return true;
    }
    if (!ContainOptionalInput(pseMode, hasAtten, hasDrop)) {
        if (!isS2Base64 || isFp32) {
            return true;
        }
    }
    return false;
}

__aicore__ constexpr TPosition GetC2Position(DTemplateType dTemplateType, bool ubOutCondition, bool isNdS2Size256) {
    if ((uint16_t)dTemplateType <= (uint16_t)DTemplateType::Aligned128 ||
        (ubOutCondition && (uint16_t)dTemplateType <= (uint16_t)DTemplateType::Aligned192) ||
        isNdS2Size256) {
        return TPosition::VECCALC;
    } else {
        return TPosition::GM;
    }
}

template <typename ChildClass, typename INPUT_T, typename T = INPUT_T,
    ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION,
    LayOutTypeEnum layout = LayOutTypeEnum::None,
    S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
    DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false,
    typename OUTPUT_T = INPUT_T, bool isInfer = false, bool isPa = false, bool isFd = false>
class FlashAttentionScoreS1s2Const {
public:
    __aicore__ inline FlashAttentionScoreS1s2Const() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, 
                                __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, 
                                __gm__ uint8_t *deqScaleV, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace,
                                const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T>;
    constexpr static MatmulConfig mm1Cfg = GetMm1Cfg(IsSameType<INPUT_T, float>::value, s1TemplateType, s2TemplateType,
        dTemplateType, pseMode, hasAtten, hasDrop);
    constexpr static auto stcMm1Cfg = matmul::GetMatmulApiTiling<a1Type, b1Type, c1Type, bias1Type>(mm1Cfg);

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;

    // define pse datatype
    using pseShiftType = typename AscendC::Conditional<POST_QUANT, INPUT_T, OUTPUT_T>::type;

    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, stcMm1Cfg, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        Bmm1ConstPolicySelector<IsSameType<INPUT_T, float>::value, dTemplateType, s2TemplateType, s1TemplateType,
            ContainOptionalInput(pseMode, hasAtten, hasDrop) || IsSameType<INPUT_T, fp8_e5m2_t>::value ||
            IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value,
            isPa, hasRope>::template Result> bmm1;

    // define matmul2
    // Dn场景的a2是[s2,s1]，需要转置
    using a2Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, INPUT_T,
        IsDn((IsSameType<INPUT_T, float>::value || IsSameType<INPUT_T, fp8_e5m2_t>::value ||
            IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value),
            pseMode, hasAtten, hasDrop,
            s1TemplateType == S1TemplateType::Aligned64, dTemplateType, hasRope),
        LayoutMode::NONE, true, TPosition::VECOUT>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<GetC2Position(dVTemplateType,
        UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
                                s1TemplateType == S1TemplateType::Aligned64),
        (s2TemplateType == S2TemplateType::Aligned256 && s1TemplateType == S1TemplateType::Aligned64)),
        CubeFormat::ND, T>;
    constexpr static MatmulConfig mm2Cfg = GetMm2Cfg(IsSameType<INPUT_T, float>::value, s1TemplateType, s2TemplateType,
        dTemplateType, dVTemplateType, pseMode, hasAtten, hasDrop);
    constexpr static auto stcMm2Cfg = matmul::GetMatmulApiTiling<a2Type, b2Type, c2Type, bias2Type>(mm2Cfg);
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, stcMm2Cfg, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        Bmm2ConstPolicySelector<IsSameType<INPUT_T, float>::value, dVTemplateType, s2TemplateType, s1TemplateType,
            (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
            IsSameType<INPUT_T, hifloat8_t>::value), isPa>::template Result> bmm2;

    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix,
                                     __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *deqScaleQ,
                                     __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *queryRope,
                                     __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable,
                                     __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                     __gm__ uint8_t *workspace,
                                     const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void GetQueryOffset(RunParamStr<isInfer> &runParam);
    __aicore__ inline int64_t GetQueryRopeOffset(RunInfo<isInfer> &runParam);
    __aicore__ inline void GetKeyOffset(RunInfo<isInfer> &runInfo);
    __aicore__ inline int64_t GetKeyRopeOffset(RunInfo<isInfer> &runInfo);
    __aicore__ inline void GetValueOffset(RunInfo<isInfer> &runInfo, int64_t &currentValueOffset);
    __aicore__ inline void Bmm1SetTensorA(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void Bmm1SetTensorB(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void IterateBmm1(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, FAFlagDataWhole<hasRope> &flag, bool isLast);
    __aicore__ inline void IterateBmm1WithRope(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, FAFlagDataWhole<hasRope> &flag, bool isLast);
    __aicore__ inline void WaitBmm1Result();
    __aicore__ inline void IterateBmm2(RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor);
    __aicore__ inline void WaitBmm2Result();
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar,
                                                   SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);
    __aicore__ inline void BroadCastAndCopyOut(RunInfo<isInfer> &runInfo, GlobalTensor<T> &sumGm,
                                               GlobalTensor<T> &maxGm, int64_t gmOffset, int64_t calculateSize);
    __aicore__ inline void ProcessVec1(RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor);
    __aicore__ inline void ProcessVec1Dn(RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor);
    __aicore__ inline void ProcessVec1Nd(RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor);
    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);

    __aicore__ inline void ProcessVec2OnUb(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo, LocalTensor<T> &bmm2Ub, int64_t vec2CalcSize);
    __aicore__ inline void ProcessVec2DSplit(RunInfo<isInfer> &runInfo);
    __aicore__ inline void ProcessVec2(RunInfo<isInfer> &runInfo);

    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = INPUT_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                           int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);

    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
    TPipe *pipe;

    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    /* =====================GM变量==================== */
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<pseShiftType> pseGm;
    __gm__ uint8_t *pseSlope;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<T> bmm2ResGm[3];
    GlobalTensor<T> vec2ResGm[3];
    GlobalTensor<float> deScaleQGm;
    GlobalTensor<float> deScaleKGm;
    GlobalTensor<float> deScaleVGm;
    GlobalTensor<INPUT_T> queryRopeGm;
    GlobalTensor<INPUT_T> keyRopeGm;
    GM_ADDR prefixNAddr;

    /* =====================UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TQue<QuePosition::VECIN, 1> pseInQue;
    TBuf<> bmm1ResBuf[2];
    TBuf<> bmm2ResBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> vselrIndexesBuf[4];
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    /* Vector1结果存放在L1上，使用Tscm变量来存储 */
    TSCM<QuePosition::VECIN, 1, 0x4> scm[2];
    /* =====================核Index信息==================== */
    int32_t blockIdx;
    /* =================编译期常量的基本块信息++=============== */
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyCountDn = s1BaseSize >> 5;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = s1BaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = s1BaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = s1BaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
                                        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten && hasDrop;
    T negativeFloatScalar;
    T positiveFloatScalar;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    /* 是否使能dn的信息; 没有可选输入并且S2切分的时候使用dn，s2比较小的时候nd效果更好 */
    static constexpr bool useDn = IsDn((IsSameType<INPUT_T, float>::value || isFp8), pseMode, hasAtten, hasDrop,
                                       s1BaseSize == 64, dTemplateType, hasRope);
    static constexpr bool bmm2Write2Ub = GetC2Position(dVTemplateType,
                                                       UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
                                                                               s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64)) == TPosition::VECCALC;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;

    /* ===========常量信息，只和输入shape相关的信息============= */
    ConstInfo<isInfer, hasRope> constInfo;
    PseInfo pseInfo;
    AttenMaskInfo attenMaskInfo;
    /* =================其他正向独有的一些信息================= */
    // KeyOffset记录，value总是可以使用上一次Key的Offset
    int64_t keyRopeOffset[3];
    FAFlagData keyFlag[3] = {};
    // MlaFAFlagData keyRopeFlag[3] = {};
    // Bmm2阶段subblock在Gm上的偏移
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    bool softMaxCheckRes = true;

    // Unpack参数
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
};

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    constInfo.subBlockIdx = get_subblockid();
    this->pipe = tPipe;
    this->tilingData = tiling;
    GetDerived()->InitUniqueOutput(softmaxLse, attentionOut);
    this->ComputeConstexpr();
    this->InitInput(query, key, value, pse, dropMask, paddingMask, attenMask, prefix,
                    actualSeqLengths, actualSeqLengthsKv, deqScaleQ, deqScaleK, deqScaleV, postQuantScale, postQuantOffset,
                    queryRope, keyRope, blockTable, queryPaddingSize, kvPaddingSize, softmaxMax, softmaxSum, softmaxOut,
                    workspace, tiling, tPipe); // gm设置

    this->InitBuffer();
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    if constexpr (IsSameType<T, float>::value) {
        uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
        negativeScalar = *((float *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
                positiveScalar = *((float *)&tmp2);
            }
        }
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
        negativeScalar = *((half *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
                positiveScalar = *((half *)&tmp2);
            }
        }
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::InitInput(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *deqScaleQ,
    __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, 
    __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    if constexpr (hasPse) {
        this->pseGm.SetGlobalBuffer((__gm__ pseShiftType *)pse);
        this->pseSlope = pse;
    }
    if constexpr (isFp8) {
        this->deScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
        this->deScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
        this->deScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
    }

    if constexpr (hasRope) {
        this->queryRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
        this->keyRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }
    if constexpr (hasAtten) {
        this->prefixNAddr = prefix;
        this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    }

    // 这里Unique放前面是因为workspace偏移计算有依赖
    GetDerived()->InitUniqueInput(key, value, dropMask, softmaxMax, softmaxSum, actualSeqLengths, actualSeqLengthsKv,
        blockTable, queryPaddingSize, kvPaddingSize, postQuantScale, postQuantOffset, workspace);

    int64_t bmm2ResBlock = this->tilingData->inputParamsRegbase.dSizeV;
    if constexpr (splitD) {
        bmm2ResBlock = (int64_t)dVTemplateType;
    }
    if constexpr (!bmm2Write2Ub) {
        int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小， Gm上的数据按照实际的dSize存储
        int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * sizeof(T);
        int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
        int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
        int64_t totalOffset = (this->blockIdx >> 1) * 3 * mm2Offset;
        if constexpr (splitD) {
            totalOffset = (this->blockIdx >> 1) * 3 * (mm2Offset + vec2Offset);
        }
        // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
        this->bmm2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
        this->bmm2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset));
        this->bmm2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset * 2));
        if constexpr (splitD) {
            this->vec2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset * 3));
            this->vec2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset * 3 + vec2Offset));
            this->vec2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mm2Offset * 3 + vec2Offset * 2));
            vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
        }
        bmm2SubBlockOffset = constInfo.subBlockIdx * mm2ResultSize >> 1; // s1BaseSize一定可以被2整除
   }
   this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    this->pipe->InitBuffer(this->softmaxSumBuf[0], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxSumBuf[1], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxSumBuf[2], 256); // [64, 1]
    this->pipe->InitBuffer(this->maxBrdcst, 1, 2048); // [64, 8]
    this->pipe->InitBuffer(this->sumBrdcst, 1, 2048); // [64, 8]
    this->pipe->InitBuffer(this->softmaxMaxBuf[0], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxMaxBuf[1], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxMaxBuf[2], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxExpBuf[0], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxExpBuf[1], 256); // [64, 1]
    this->pipe->InitBuffer(this->softmaxExpBuf[2], 256); // [64, 1]
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::InitBuffer()
{
    if constexpr (s2BaseSize == 256) {
        if constexpr (s1BaseSize == 128) {
            this->pipe->InitBuffer(this->bmm2ResBuf[0], 64 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->bmm2ResBuf[1], 64 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            SoftmaxInitBuffer();
            this->pipe->InitBuffer(this->stage1OutQue[0], 1, 33024);
            this->pipe->InitBuffer(this->bmm1ResBuf[0], 65536);
            this->pipe->InitBuffer(this->bmm1ResBuf[1], 65536);
        } else {
            SoftmaxInitBuffer();
            this->pipe->InitBuffer(this->commonTBuf, 512); // 实际上只需要512Bytes
            this->pipe->InitBuffer(this->bmm2ResBuf[0], 32 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->bmm2ResBuf[1], 32 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->stage1OutQue[0], 1, 16896);
            this->pipe->InitBuffer(this->stage1OutQue[1], 1, 16896);
            this->pipe->InitBuffer(this->bmm1ResBuf[0], 32768);
            this->pipe->InitBuffer(this->bmm1ResBuf[1], 32768);
            if constexpr (hasAtten) {
                this->pipe->InitBuffer(this->attenMaskInQue[0], 1, 8192);
                this->pipe->InitBuffer(this->attenMaskInQue[1], 1, 8192);
            }
            if constexpr (hasPseOuter) {
                this->pipe->InitBuffer(this->pseInQue, 1, 16384);
            }
        }
    } else {
        SoftmaxInitBuffer();
        if constexpr (!useDn) {
            if constexpr (hasPseOuter) {
                if constexpr (IsSameType<INPUT_T, float>::value) {
                    this->pipe->InitBuffer(this->pseInQue, 1, 32768);
                } else {
                    this->pipe->InitBuffer(this->pseInQue, 1, 16384);
                }
            }

            if constexpr (hasAtten) {
                this->pipe->InitBuffer(this->attenMaskInQue[0], 1, 8192);
                this->pipe->InitBuffer(this->attenMaskInQue[1], 1, 8192);
            }

            if constexpr (!IsSameType<INPUT_T, float>::value) {
                this->pipe->InitBuffer(this->commonTBuf, 512); // 实际上只需要512Bytes
            }
        }
        if constexpr (bmm2Write2Ub) {
            // 小于128Bmm2结果和Vec2结果都在UB
            this->pipe->InitBuffer(this->bmm2ResBuf[0], 64 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->bmm2ResBuf[1], 64 * dTemplateAlign64 * sizeof(T));
            this->pipe->InitBuffer(this->stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else if constexpr (dTemplateAlign64 <= 256) {
            // bmm2结果在Gm，Vector2结果在UB，开启多层循环，每次处理32KB
            this->pipe->InitBuffer(this->bmm2ResBuf[0], 32768);
            this->pipe->InitBuffer(this->stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        } else {
            // bmm2结果在Gm，Vector2结果也在Gm，开启多层循环，每次处理32KB
            this->pipe->InitBuffer(this->bmm2ResBuf[0], 32768);
            this->pipe->InitBuffer(this->stage2OutBuf, 32768);
        }
        if constexpr (IsSameType<INPUT_T, float>::value) {
            this->pipe->InitBuffer(this->stage1OutQue[0], 1, 33280);
        } else if constexpr (isFp8) {
            this->pipe->InitBuffer(this->stage1OutQue[0], 1, 8320);
            this->pipe->InitBuffer(this->stage1OutQue[1], 1, 8320);
        } else {
            this->pipe->InitBuffer(this->stage1OutQue[0], 1, 16640);
            this->pipe->InitBuffer(this->stage1OutQue[1], 1, 16640);
        }
        this->pipe->InitBuffer(this->bmm1ResBuf[0], 32768);
        this->pipe->InitBuffer(this->bmm1ResBuf[1], 32768);
    }
    if constexpr (isFp8) {
        this->pipe->InitBuffer(this->vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128); // s2realsize (64, 128]
        this->pipe->InitBuffer(this->vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);  // s2realsize (0, 64]

        LocalTensor<uint8_t> vselrIndexesBuf =
            this->vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 128; i++) {
            vselrIndexesBuf.SetValue(i, i * 2); 
        }
        vselrIndexesBuf =
            this->vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 64; i++) {
            vselrIndexesBuf.SetValue(i, i * 4); 
        }
    }
    GetDerived()->InitUniqueLocalBuffer();

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    // 计算轴的乘积
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;

    constInfo.n2Size = inputParamsRegbase.n2Size;
    constInfo.s1Size = inputParamsRegbase.s1Size;
    constInfo.s2Size = inputParamsRegbase.s2Size;
    constInfo.dSize = inputParamsRegbase.dSize;
    constInfo.dSizeV = inputParamsRegbase.dSizeV;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    if constexpr (hasRope) {
        constInfo.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = inputParamsRegbase.gSize;
    constInfo.s1OuterSize = this->tilingData->multiCoreParamsRegbase.s1OuterSize;
    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.s2D = constInfo.s2Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2D = constInfo.n2Size * constInfo.dSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    int64_t bSize = inputParamsRegbase.bSize;
    constInfo.bN2D = bSize * constInfo.n2D;
    constInfo.gS1D = constInfo.gSize * constInfo.s1D;
    constInfo.n2S2D = constInfo.n2Size * constInfo.s2D;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.bN2GD = bSize * constInfo.n2GD;
    constInfo.n2GS1D = constInfo.n2Size * constInfo.gS1D;
    // 计算切分轴的乘积
    constInfo.s2BaseN2D = s2BaseSize * constInfo.n2D;
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
        constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
        constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
        constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
        constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
        constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
        constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
        constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
        constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    } else {
        constInfo.s1Dv = constInfo.s1D;
        constInfo.s2Dv = constInfo.s2D;
        constInfo.n2Dv = constInfo.n2D;
        constInfo.gDv = constInfo.gD;
        constInfo.gS1Dv = constInfo.gS1D;
        constInfo.n2S2Dv = constInfo.n2S2D;
        constInfo.n2GDv = constInfo.n2GD;
        constInfo.s2BaseN2Dv = constInfo.s2BaseN2D;
        constInfo.n2GS1Dv = constInfo.n2GS1D;
    }
    uint32_t headNumRatio = inputParamsRegbase.headNumRatio;
    constInfo.layoutType = inputParamsRegbase.layoutType;
    constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);

    if constexpr (hasRope) {
        constInfo.s1DR = constInfo.s1Size * constInfo.dSizeRope;
        constInfo.s2DR = constInfo.s2Size * constInfo.dSizeRope;
        constInfo.gDR = constInfo.gSize * constInfo.dSizeRope;
        constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
        constInfo.bN2DR = bSize * constInfo.n2DR;
        constInfo.gS1DR = constInfo.gSize * constInfo.s1DR;
        constInfo.n2S2DR = constInfo.n2Size * constInfo.s2DR;
        constInfo.n2GDR = constInfo.n2Size * constInfo.gDR;
        constInfo.bN2GDR = bSize * constInfo.n2GDR;
        constInfo.n2GS1DR = constInfo.n2Size * constInfo.gS1DR;
        constInfo.s2BaseN2DR = s2BaseSize * constInfo.n2DR;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
        }
        constInfo.mm1Ka = constInfo.n2GD;
        constInfo.mm1Kb = constInfo.n2D;
        constInfo.mm2Kb = constInfo.n2Dv;
        constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        if constexpr (isInfer) {
            if (inputParamsRegbase.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
                constInfo.attentionOutStride = 0;
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
            constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
            if constexpr (hasRope) {
                constInfo.s1BaseN2GDR = s1BaseSize * constInfo.n2GDR;
            }
            constInfo.mm1Ka = constInfo.n2GD;
            constInfo.mm1Kb = constInfo.n2D;
            constInfo.mm2Kb = constInfo.n2Dv;
            constInfo.attentionOutStride =
                (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (inputParamsRegbase.isGqa) {
                    constInfo.mm1Ka = constInfo.dSize;
                    constInfo.attentionOutStride = 0;
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            constInfo.s1BaseBN2GD = s1BaseSize * constInfo.bN2GD;
            constInfo.s2BaseBN2D = bSize * constInfo.s2BaseN2D;
            constInfo.bN2GDv = bSize * constInfo.n2GDv;
            constInfo.s1BaseBN2GDv = s1BaseSize * constInfo.bN2GDv;
            constInfo.s2BaseBN2Dv = bSize * constInfo.s2BaseN2Dv;
            if constexpr (hasRope) {
                constInfo.s1BaseBN2GDR = s1BaseSize * constInfo.bN2GDR;
                constInfo.s2BaseBN2DR = bSize * constInfo.s2BaseN2DR;
            }
            constInfo.mm1Ka = constInfo.bN2GD;
            constInfo.mm1Kb = constInfo.bN2D;
            constInfo.mm2Kb = bSize * constInfo.n2Dv;
            constInfo.attentionOutStride =
                (bSize * constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            constInfo.s1BaseD = s1BaseSize * constInfo.dSize;
            constInfo.s2BaseD = s2BaseSize * constInfo.dSize;
            constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
            constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
            if constexpr (hasRope) {
                constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
                constInfo.s2BaseDR = s2BaseSize * constInfo.dSizeRope;
            }
            constInfo.mm1Ka = constInfo.dSize;
            constInfo.mm1Kb = constInfo.dSize;
            constInfo.mm2Kb = constInfo.dSizeV;
            constInfo.attentionOutStride = 0;
        }
    }

    if constexpr (hasPse == true) {
        this->pseInfo.pseLayoutType = inputParamsRegbase.pseShapeType;
        this->pseInfo.pseType = inputParamsRegbase.pseType;
        this->pseInfo.pseBSize = inputParamsRegbase.pseBSize;
        this->pseInfo.pseS1Size = inputParamsRegbase.pseS1Size;
        this->pseInfo.pseS2Size = inputParamsRegbase.pseS2Size;
        this->pseInfo.pseEncodeType = (uint32_t)inputParamsRegbase.pseEncodeType;
        this->pseInfo.pseStride = pseInfo.pseLayoutType == pse1S2 ? 0 : s2BaseSize;
        this->pseInfo.qStartIdx = inputParamsRegbase.qStartIdx;
        this->pseInfo.kvStartIdx = inputParamsRegbase.kvStartIdx;
        if (inputParamsRegbase.pseShapeType == pse1S2) {
            constInfo.gS2 = constInfo.gSize * constInfo.s2Size;
        }
    }

    constInfo.matmulMSize = constInfo.s1Size;
    if constexpr (isInfer) {
        // GS1合轴
        if (inputParamsRegbase.isGqa) {
            constInfo.matmulMSize = constInfo.gS1;
        }
    }

    if constexpr (hasAtten == true) {
        this->attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
        this->attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
        this->attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
        this->attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
        this->attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
        this->attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
        this->attenMaskInfo.prefixNAddr = prefixNAddr;
        this->attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
    }

    // if hasRope, kfc-message is larger than 8 bytes, should call mm's func in order
    if constexpr (!hasRope) {
        if constexpr (!useDn) {
            this->bmm1.SetOrgShape(constInfo.matmulMSize, constInfo.s2Size, constInfo.mm1Ka,
                constInfo.mm1Kb, s2BaseSize);
        } else {
            this->bmm1.SetOrgShape(constInfo.s2Size, constInfo.matmulMSize, constInfo.mm1Kb,
                constInfo.mm1Ka, s1BaseSize);
        }
    }
    this->bmm2.SetOrgShape(constInfo.matmulMSize, constInfo.mm2Kb, s2BaseSize, constInfo.mm2Kb, dTemplateAlign64);
    GetDerived()->InitUniqueConstInfo(inputParamsRegbase);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::Process()
{
    GetDerived()->Process();
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
    int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam)
{
    // 计算轴的idx
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
        int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);

        while (multiCoreInnerIdx >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += runParam.actualS1Size;
            this->s2SizeAcc += runParam.actualS2Size;
            runParam.b1SSOffset += runParam.actualS1Size * runParam.actualS2Size;
            if (hasDrop) {
                runParam.b1SSOffsetAlign16 += runParam.actualS1Size * Align(runParam.actualS2Size);
            }
            runParam.boIdx++;
            if (runParam.boIdx >= this->tilingData->inputParamsRegbase.bSize) {
                break;
            }
            GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
            actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);
        }

        int64_t tmpS1Outersize = CeilDiv(runParam.actualS1Size, this->s1BaseSize);
        actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
        runParam.n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParamsRegbase.gSize;
        runParam.goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParamsRegbase.gSize;
        runParam.s1oIdx = actualS1Outersize % tmpS1Outersize;
    } else {
        runParam.boIdx = multiCoreInnerIdx / constInfo.n2GS1o;
        runParam.n2oIdx = multiCoreInnerIdx % constInfo.n2GS1o / constInfo.gS1o;
        runParam.goIdx = multiCoreInnerIdx % constInfo.gS1o / constInfo.s1OuterSize;
        runParam.s1oIdx = multiCoreInnerIdx % constInfo.s1OuterSize;
        runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
        runParam.actualS1Size = constInfo.s1Size;
        runParam.actualS2Size = constInfo.s2Size;
        if (hasDrop) {
            runParam.b1SSOffsetAlign16 = runParam.boIdx * constInfo.s1Size * Align(constInfo.s2Size);
        }
    }
    runParam.s1RealSize = Min(s1BaseSize, runParam.actualS1Size - runParam.s1oIdx * s1BaseSize);
    if constexpr (useDn) {
        runParam.s1RealSizeAlign32 = (runParam.s1RealSize + 31) >> 5 << 5;
        runParam.halfS1RealSize = runParam.s1RealSize <= 16 ? runParam.s1RealSize : (runParam.s1RealSizeAlign32 >> 1);
    } else {
        runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    }
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::WaitBmm1Result()
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2LoopStartIdx = runParam.s2LoopStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (isFd) {
        runInfo.flashDecodeS2Idx = (this->blockIdx >> 1) % constInfo.splitKVNum;
    }
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    this->ComputeBmm1Tail(runInfo, runParam);
    GetDerived()->InitUniqueRunInfo(runParam, runInfo);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;  // D>128 这里需要适配

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if constexpr (isInfer) {
        if ((runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    } else {
        if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::IterateBmm1(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, FAFlagDataWhole<hasRope> &flag, bool isLast)
{
    if constexpr (useDn) {
        // 硬件限制，splitN场景需要32对齐
        this->bmm1.SetTail(runInfo.s2RealSize, runInfo.s1RealSizeAlign32, constInfo.dSize);
        flag.baseFlag.mOrNAdditionalSize = runInfo.s1RealSizeAlign32 - runInfo.s1RealSize;
    } else {
        // splitM场景mm内部2对齐
        this->bmm1.SetTail(runInfo.s1RealSize, runInfo.s2RealSize, constInfo.dSize);
    }

    this->Bmm1SetTensorA(runInfo, runParam);
    this->Bmm1SetTensorB(runInfo, runParam);
    flag.baseFlag.copyCurrent = 1;
    flag.baseFlag.copyNext = 0;

    if constexpr (useDn) {
        flag.baseFlag.leftBufIdx = runInfo.taskIdMod2; // 0/1
        flag.baseFlag.rightBufIdx = 4 + runInfo.multiCoreIdxMod2; // 4/5
        this->bmm1.SetSelfDefineData(flag);
        flag.baseFlag.reuseRight = 1;
    } else {
        flag.baseFlag.leftBufIdx = runInfo.multiCoreIdxMod2;
        flag.baseFlag.rightBufIdx = 2 + runInfo.taskIdMod2; // 0/1
        this->bmm1.SetSelfDefineData(flag);
        flag.baseFlag.reuseLeft = 1;
    }

    keyFlag[runInfo.taskIdMod3] = flag.baseFlag;
    LocalTensor<T> stage1PongTensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    this->bmm1.template IterateAll<false>(stage1PongTensor, 0, true, true);
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::IterateBmm1WithRope(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, FAFlagDataWhole<hasRope> &mlaFlag, bool isLast)
{
    GetKeyOffset(runInfo);
    int64_t queryRopeOffset = GetQueryRopeOffset(runInfo);
    keyRopeOffset[runInfo.taskIdMod3] = GetKeyRopeOffset(runInfo);
    mlaFlag.qRopeAddr = (uint64_t)this->queryRopeGm[queryRopeOffset].GetPhyAddr();
    mlaFlag.kRopeAddr = (uint64_t)this->keyRopeGm[keyRopeOffset[runInfo.taskIdMod3]].GetPhyAddr();

    mlaFlag.baseFlag.copyCurrent = 1;
    mlaFlag.baseFlag.copyNext = 0;
    mlaFlag.baseFlag.leftBufIdx = runInfo.multiCoreIdxMod2;
    mlaFlag.baseFlag.rightBufIdx = 2 + runInfo.taskIdMod2; // 0/1
    this->bmm1.SetSelfDefineData(mlaFlag);
    mlaFlag.baseFlag.reuseLeft = 1;
    keyFlag[runInfo.taskIdMod3] = mlaFlag.baseFlag;

    this->bmm1.SetOrgShape(constInfo.matmulMSize, constInfo.s2Size, constInfo.mm1Ka,
        constInfo.mm1Kb, s2BaseSize);
    this->bmm1.SetTail(runInfo.s1RealSize, runInfo.s2RealSize,
        (constInfo.dSize + constInfo.dSizeRope));
    this->bmm1.SetTensorA(this->queryGm[runParam.tensorQOffset]);
    this->bmm1.SetTensorB(GetDerived()->GetKeyGm(runInfo)[runInfo.keyOffset], true);
    LocalTensor<T> stage1PongTensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    this->bmm1.template IterateAll<false>(stage1PongTensor, 0, true, true);
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetQueryOffset(
    RunParamStr<isInfer> &runParam)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;

    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t bOffsetOut = 0;
    int64_t s1OffsetOut = 0;
    int64_t n2OffsetOut = 0;
    int64_t gOffsetOut = 0;
    int64_t subBlockS1Offset = 0;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffset = this->s1SizeAcc * constInfo.n2GD;
        s1Offset = runParam.s1oIdx * constInfo.s1BaseN2GD;
        n2Offset = runParam.n2oIdx * constInfo.gD;
        gOffset = runParam.goIdx * constInfo.dSize;
        bOffsetOut = this->s1SizeAcc * constInfo.n2GDv;
        s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseN2GDv;
        n2OffsetOut = runParam.n2oIdx * constInfo.gDv;
        gOffsetOut = runParam.goIdx * constInfo.dSizeV;
        subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.n2GDv;
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffset = runParam.boIdx * constInfo.n2GS1D;
            s1Offset = runParam.s1oIdx * constInfo.s1BaseN2GD;
            n2Offset = runParam.n2oIdx * constInfo.gD;
            gOffset = runParam.goIdx * constInfo.dSize;
            bOffsetOut = runParam.boIdx * constInfo.n2GS1Dv;
            s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseN2GDv;
            n2OffsetOut = runParam.n2oIdx * constInfo.gDv;
            gOffsetOut = runParam.goIdx * constInfo.dSizeV;
            subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.n2GDv;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            // SBH/SBNGD
            s1Offset = runParam.s1oIdx * constInfo.s1BaseBN2GD;
            bOffset = runParam.boIdx * constInfo.n2GD;
            n2Offset = runParam.n2oIdx * constInfo.gD;
            gOffset = runParam.goIdx * constInfo.dSize;
            s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseBN2GDv;
            bOffsetOut = runParam.boIdx * constInfo.n2GDv;
            n2OffsetOut = runParam.n2oIdx * constInfo.gDv;
            gOffsetOut = runParam.goIdx * constInfo.dSizeV;
            subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.bN2GDv;
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffset = runParam.boIdx * constInfo.n2GS1D;
            n2Offset = runParam.n2oIdx * constInfo.gS1D;
            gOffset = runParam.goIdx * constInfo.s1D;
            s1Offset = runParam.s1oIdx * constInfo.s1BaseD;
            bOffsetOut = runParam.boIdx * constInfo.n2GS1Dv;
            n2OffsetOut = runParam.n2oIdx * constInfo.gS1Dv;
            gOffsetOut = runParam.goIdx * constInfo.s1Dv;
            s1OffsetOut = runParam.s1oIdx * constInfo.s1BaseDv;
            subBlockS1Offset = constInfo.subBlockIdx * runParam.firstHalfS1RealSize * constInfo.dSizeV;
        }
    }
    runParam.attentionOutOffset = bOffsetOut + n2OffsetOut + gOffsetOut + s1OffsetOut + subBlockS1Offset;
    runParam.tensorQOffset = bOffset + n2Offset + gOffset + s1Offset;
}

S1S2_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetQueryRopeOffset(
    RunInfo<isInfer> &runInfo)
{
    if constexpr (isInfer) {
        return runInfo.qRopeOffset;
    } else {
        // 计算gm上的offset
        int64_t bOffsetRope = 0;
        // s1需要考虑inner轴的影响
        int64_t s1OffsetRope = 0;
        int64_t n2OffsetRope = 0;
        int64_t gOffsetRope = 0;

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffsetRope = runInfo.s1SizeAcc * constInfo.n2GDR;
            s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseN2GDR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
            gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
        } else {
            if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                // BSH/BSNGD
                bOffsetRope = runInfo.boIdx * constInfo.n2GS1DR;
                s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseN2GDR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
                gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                // SBH/SBNGD
                s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseBN2GDR;
                bOffsetRope = runInfo.boIdx * constInfo.n2GDR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
                gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                // bnsd
                bOffsetRope = runInfo.boIdx * constInfo.n2GS1DR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.gS1DR;
                gOffsetRope = runInfo.goIdx * constInfo.s1DR;
                s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
            }
        }
        return bOffsetRope + n2OffsetRope + gOffsetRope + s1OffsetRope;
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetKeyOffset(
    RunInfo<isInfer> &runInfo)
{
    if constexpr (!isInfer) {
        // 计算gm上的offset
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s2Offset = 0;

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffset = runInfo.s2SizeAcc * constInfo.n2D;
            s2Offset = runInfo.s2StartIdx * constInfo.n2D + runInfo.s2LoopCount * constInfo.s2BaseN2D;
            n2Offset = runInfo.n2oIdx * constInfo.dSize;
        } else {
            if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                // BSH/BSND
                bOffset = runInfo.boIdx * constInfo.n2S2D;
                s2Offset = runInfo.s2StartIdx * constInfo.n2D + runInfo.s2LoopCount * constInfo.s2BaseN2D;
                n2Offset = runInfo.n2oIdx * constInfo.dSize;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                // SBH/SBND
                s2Offset = runInfo.s2StartIdx * constInfo.bN2D + runInfo.s2LoopCount * constInfo.s2BaseBN2D;
                bOffset = runInfo.boIdx * constInfo.n2D;
                n2Offset = runInfo.n2oIdx * constInfo.dSize;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                // BNSD
                bOffset = runInfo.boIdx * constInfo.n2S2D;
                n2Offset = runInfo.n2oIdx * constInfo.s2D;
                s2Offset = runInfo.s2StartIdx * constInfo.dSize + runInfo.s2LoopCount * constInfo.s2BaseD;
            }
        }
        runInfo.keyOffset = bOffset + n2Offset + s2Offset;
        runInfo.valueOffset = runInfo.keyOffset;
    }
}

S1S2_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetKeyRopeOffset(
    RunInfo<isInfer> &runInfo)
{
    if constexpr (isInfer) {
        return runInfo.kRopeOffset;
    } else {
        // 计算gm上的offset
        int64_t bOffsetRope = 0;
        int64_t n2OffsetRope = 0;
        int64_t s2OffsetRope = 0;

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffsetRope = runInfo.s2SizeAcc * constInfo.n2DR;
            s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + runInfo.s2LoopCount * constInfo.s2BaseN2DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
        } else {
            if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                // BSH/BSND
                bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
                s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + runInfo.s2LoopCount * constInfo.s2BaseN2DR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                // SBH/SBND
                s2OffsetRope = runInfo.s2StartIdx * constInfo.bN2DR + runInfo.s2LoopCount * constInfo.s2BaseBN2DR;
                bOffsetRope = runInfo.boIdx * constInfo.n2DR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                // BNSD
                bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
                n2OffsetRope = runInfo.n2oIdx * constInfo.s2DR;
                s2OffsetRope = runInfo.s2StartIdx * constInfo.dSizeRope +
                    runInfo.s2LoopCount * constInfo.s2BaseDR;
            }
        }
        return bOffsetRope + n2OffsetRope + s2OffsetRope;
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::Bmm1SetTensorA(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    if constexpr (useDn) {
        // dn场景下左矩阵是Key
        GetKeyOffset(runInfo);
        this->bmm1.SetTensorA(GetDerived()->GetKeyGm(runInfo)[runInfo.keyOffset]);
    } else {
        this->bmm1.SetTensorA(this->queryGm[runParam.tensorQOffset]);
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::Bmm1SetTensorB(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    if constexpr (useDn) {
        this->bmm1.SetTensorB(this->queryGm[runParam.tensorQOffset], true);
    } else {
        GetKeyOffset(runInfo);
        this->bmm1.SetTensorB(GetDerived()->GetKeyGm(runInfo)[runInfo.keyOffset], true);
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec1(
    RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor)
{
    if constexpr (useDn) {
        ProcessVec1Dn(runInfo, scmTensor);
    } else {
        ProcessVec1Nd(runInfo, scmTensor);
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec1Dn(
    RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor)
{
    LocalTensor<T> stage1PongTensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];

    auto expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[0];
    int64_t stage1Offset = 0;
    if constexpr (s2BaseSize == 128) {
        stage1Offset = runInfo.taskIdMod2;
    }
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
        fa::ProcessVec1VfDn<T, INPUT_T, false, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, stage1PongTensor, expUb,
            runInfo.s1RealSizeAlign32 >> 1, runInfo.s2AlignedSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue),
            negativeFloatScalar, this->constInfo.keepProb);
    } else {
        fa::ProcessVec1VfDn<T, INPUT_T, true, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, stage1PongTensor, expUb,
            runInfo.s1RealSizeAlign32 >> 1, runInfo.s2AlignedSize, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue),
            negativeFloatScalar, this->constInfo.keepProb);
    }

    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    //-------------------------Data copy to l1-------------------------
    scmTensor = scm[runInfo.taskIdMod2].template AllocTensor<INPUT_T>();

    if (runInfo.s2RealSize > vec1S2CopyLenDn) {
        DataCopy(scmTensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.s2AlignedSize], stage1CastTensor,
            {vec1S2CopyCountDn, vec1S2CopyLenDn, 1, static_cast<uint16_t>(runInfo.s2AlignedSize - vec1S2CopyLenDn)});
        DataCopy(scmTensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.s2AlignedSize + vec1S2strideDn],
            stage1CastTensor[vec1ResOffsetDn],
            {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.s2AlignedSize - vec1S2CopyLenDn),
            static_cast<uint16_t>(s2BaseSize - runInfo.s2AlignedSize + 1), vec1S2CopyLenDn});
    } else {
        DataCopy(scmTensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.s2AlignedSize], stage1CastTensor,
            {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.s2AlignedSize),
            static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.s2AlignedSize + 1), 0});
    }
    scm[runInfo.taskIdMod2].EnQue(scmTensor);
    scm[runInfo.taskIdMod2].template DeQue<INPUT_T>();
    //-----------------------------------------------------------------
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        GetDerived()->SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
    }
    return;
}

S1S2_TEMPLATE
__aicore__ inline bool FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::SoftmaxInvalidLineCheck(
    LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
        T maxValue = maxUb.GetValue(i);
        uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
        if (checkValue == negativeIntScalar) {
            isUpdateNeedCheck = true;
            break;
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
{
    if (this->softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VAULE_FP32, softmaxShapeInfo);
        if (!res) {
            this->softMaxCheckRes = false;
        } else {
            if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::BroadCastAndCopyOut(
    RunInfo<isInfer> &runInfo, GlobalTensor<T> &sumGm,
    GlobalTensor<T> &maxGm, int64_t gmOffset, int64_t calculateSize)
{
    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
    fa::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.template EnQue(sumOutTensor);
    sumBrdcst.template DeQue<float>();
    DataCopy(sumGm[gmOffset], sumOutTensor, calculateSize);
    sumBrdcst.template FreeTensor(sumOutTensor);

    // Copy max to gm
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
    fa::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
    maxBrdcst.template EnQue(maxOutTensor);
    maxBrdcst.template DeQue<float>();
    DataCopy(maxGm[gmOffset], maxOutTensor, calculateSize);
    maxBrdcst.template FreeTensor(maxOutTensor);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec1Nd(
    RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor)
{
    PseCopyIn<T, pseShiftType, hasPseOuter>(this->pseInQue, this->pseGm, runInfo, constInfo, pseInfo);
    LocalTensor<pseShiftType> pseUb;
    if constexpr (hasPseOuter == true) {
        pseUb = this->pseInQue.template DeQue<pseShiftType>();
    }
    float slopes = 0.0f;
    float posShift = 0.0f;
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            if (this->tilingData->inputParamsRegbase.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL) &&
                runInfo.boIdx != 0) {
                this->pseInfo.qStartIdx = 0;
                this->pseInfo.kvStartIdx = 0;
            }
        }
        ComputeInnerPseOffset<T, INPUT_T, hasPse>(slopes, posShift, runInfo, constInfo, pseInfo, this->pseSlope);
    }
    AttenMaskCopyIn<hasAtten, isFd>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
        this->attenMaskGmInt, runInfo, constInfo, attenMaskInfo);
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    }
    LocalTensor<uint8_t> dropMaskUb;
    GetDerived()->GenerateDropoutMask(runInfo, dropMaskUb);

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
    LocalTensor<uint8_t> apiTmpBuffer;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        apiTmpBuffer = this->sumBrdcst.template AllocTensor<uint8_t>();
    } else {
        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    }

    LocalTensor<T> stage1PongTensor = this->bmm1ResBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = 0;
    if constexpr (!IsSameType<INPUT_T, float>::value) {
        stage1Offset = runInfo.taskIdMod2;
    }
    float descaleQK = 1.0;
    if constexpr (isFp8) {
        int64_t s1BlockCnt = CeilDivision(constInfo.s1Size, FP8_QUANT_BLOCK_SIZE);
        int64_t s2BlockCnt = CeilDivision(constInfo.s2Size, FP8_QUANT_BLOCK_SIZE);
        /* Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1] */
        int64_t deScaleQOffset = runInfo.boIdx * constInfo.n2G * s1BlockCnt +
                                 runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
                                 runInfo.goIdx * s1BlockCnt + runInfo.s1oIdx;
        runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                  runInfo.n2oIdx * s2BlockCnt +
                                  (runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount;
        float deSCaleQValue = this->deScaleQGm.GetValue(deScaleQOffset);
        float deSCaleKValue = this->deScaleKGm.GetValue(runInfo.deScaleKvOffset);
        descaleQK = deSCaleQValue * deSCaleKValue;
    }

    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
        if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        } else if (runInfo.s2RealSize == 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        } else if (runInfo.s2RealSize > 128 && runInfo.s2RealSize <= 256) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        } else {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        }
    } else {
        if (runInfo.s2RealSize <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        } else if (runInfo.s2RealSize == 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        } else if (runInfo.s2RealSize > 128 && runInfo.s2RealSize <= 256) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        } else {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, stage1PongTensor, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                this->constInfo.keepProb);
        }
    }

    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    if constexpr (hasPseOuter) {
        this->pseInQue.template FreeTensor(pseUb);
    }

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    scmTensor = scm[runInfo.taskIdMod2].template AllocTensor<INPUT_T>();
    if (likely(runInfo.halfS1RealSize != 0)) {
        if constexpr (IsSameType<INPUT_T, float>::value) {
            DataCopy(scmTensor[constInfo.subBlockIdx * vec1ScmBlockFp32], stage1CastTensor,
                    {16, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
        } else if constexpr (isFp8) {
            DataCopy(scmTensor[constInfo.subBlockIdx * vec1ScmBlockFp8], stage1CastTensor,
                    {s2BaseSize / 32, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
        } else {
            DataCopy(scmTensor[constInfo.subBlockIdx * vec1ScmBlock], stage1CastTensor,
                    {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize,
                    (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
        }
    }
    scm[runInfo.taskIdMod2].EnQue(scmTensor);
    scm[runInfo.taskIdMod2].template DeQue<INPUT_T>();
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    // ======================================================
    if (runInfo.s2LoopCount != runInfo.s2LoopStartIdx) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }
    if constexpr (IsSameType<INPUT_T, float>::value) {
        this->sumBrdcst.template FreeTensor(apiTmpBuffer);
    }
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, sumUb, maxUb);
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        GetDerived()->SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
    }
    return;
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::WaitBmm2Result()
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::GetValueOffset(
    RunInfo<isInfer> &runInfo, int64_t &currentValueOffset)
{
    if constexpr (!isInfer) {
        // 计算gm上的offset
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s2Offset = 0;

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // (BS)ND
            bOffset = runInfo.s2SizeAcc * constInfo.n2Dv;
            s2Offset = runInfo.s2StartIdx * constInfo.n2Dv + runInfo.s2LoopCount * constInfo.s2BaseN2Dv;
            n2Offset = runInfo.n2oIdx * constInfo.dSizeV;
        } else {
            if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
                // BSH/BSND
                bOffset = runInfo.boIdx * constInfo.n2S2Dv;
                s2Offset = runInfo.s2StartIdx * constInfo.n2Dv + runInfo.s2LoopCount * constInfo.s2BaseN2Dv;
                n2Offset = runInfo.n2oIdx * constInfo.dSizeV;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
                // SBH/SBND
                s2Offset = runInfo.s2StartIdx * constInfo.mm2Kb + runInfo.s2LoopCount * constInfo.s2BaseBN2Dv;
                bOffset = runInfo.boIdx * constInfo.n2Dv;
                n2Offset = runInfo.n2oIdx * constInfo.dSizeV;
            } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BNSD) {
                // BNSD
                bOffset = runInfo.boIdx * constInfo.n2S2Dv;
                n2Offset = runInfo.n2oIdx * constInfo.s2Dv;
                s2Offset = runInfo.s2StartIdx * constInfo.dSizeV + runInfo.s2LoopCount * constInfo.s2BaseDv;
            }
        }
        currentValueOffset = bOffset + n2Offset + s2Offset;
    }
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::IterateBmm2(
    RunInfo<isInfer> &runInfo, LocalTensor<INPUT_T> &scmTensor)
{
    int64_t currentValueOffset = runInfo.valueOffset;
    FAFlagData flag = keyFlag[runInfo.taskIdMod3];
    // d等长时value预取的地址和当前循环的keyOffset相同，当前Bmm2的循环落后Bmm1循环一次，所以index是1 - runInfo.taskIdMod2
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetValueOffset(runInfo, currentValueOffset);
        flag.copyCurrent = 1;
        flag.copyNext = 0;
    }
    if constexpr (useDn) {
        this->bmm2.SetTensorA(scmTensor, true);
    } else {
        this->bmm2.SetTensorA(scmTensor);
    }
    scm[runInfo.taskIdMod2].FreeTensor(scmTensor);
    this->bmm2.SetTensorB(GetDerived()->GetValueGm(runInfo)[currentValueOffset]);
    this->bmm2.SetTail(-1, constInfo.dSizeV, runInfo.s2RealSize);

    /* value的BufIdx永远是Key的+2, key的index在dn场景下是左矩阵的index */
    if constexpr (useDn) {
        flag.rightBufIdx = flag.leftBufIdx + 2;
    } else {
        flag.rightBufIdx = flag.rightBufIdx + 2;
    }

    this->bmm2.SetSelfDefineData(flag);
    if constexpr (bmm2Write2Ub) {
        LocalTensor<T> bmm2Res = this->bmm2ResBuf[runInfo.taskIdMod2].template Get<T>(); // i.a 32k
        this->bmm2.template IterateAll<false>(bmm2Res, 0, true, true);
    } else {
        if constexpr (splitD) {
            this->bmm2.template IterateAll<false>(bmm2ResGm[runInfo.taskIdMod3], 0, false, true);
        } else {
            this->bmm2.template IterateAll<false>(bmm2ResGm[runInfo.taskIdMod3], 0, true, true);
        }
    }
}

S1S2_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ComputeOffsetForSoftmax(
    RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec2OnUb(RunInfo<isInfer> &runInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        return;
    }
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    LocalTensor<T> bmm2Ub = this->bmm2ResBuf[runInfo.taskIdMod2].template Get<T>();
    if constexpr (implMode != ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION && !isFp8) {
        if constexpr (isInfer) {
            if (constInfo.s2Size <= 128 && !constInfo.isRowInvalid && !POST_QUANT) { // 128: kv方向基本块大小
                ProcessVec2NoGlobalUpdate(runInfo, bmm2Ub, (dTemplateAlign64 * sizeof(INPUT_T)) << 5);
                return;
            }
        } else {
            if (constInfo.s2Size <= 128) { // 128: kv方向基本块大小
                ProcessVec2NoGlobalUpdate(runInfo, bmm2Ub, (dTemplateAlign64 * sizeof(INPUT_T)) << 5);
                return;
            }
        }
    }
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    float deSCaleVValue;
    if constexpr (isFp8) {
        deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
    }
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
        DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        float deSCalePreVValue = 1.0f;
        if constexpr (isFp8) {
            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
        }
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);
            } else {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);                
            }
        } else {
            if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);            
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, bmm2Ub, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    deSCaleVValue, deSCalePreVValue);
            }
        }
    }

    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
        }
        GetDerived()->CopyOutAttentionOut(runInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec2NoGlobalUpdate(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &bmm2Ub, int64_t vec2CalcSize) {
    LocalTensor<INPUT_T> vec2ResUb = this->stage2OutBuf.template Get<INPUT_T>()[runInfo.taskIdMod2 * vec2CalcSize];
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    DivCast<T, INPUT_T, dTemplateAlign64>(vec2ResUb, bmm2Ub, sumUb, runInfo.vec2S1RealSize);
    Bmm2DataCopyOut(runInfo, vec2ResUb, 0);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec2DSplit(RunInfo<isInfer> &runInfo) {
    // bmm2 result is on GM and global update data on UB
    runInfo.vec2S1BaseSize = 8192 / constInfo.dBasicBlock;
    int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    LocalTensor<T> bmm2Ub = this->bmm2ResBuf[0].template Get<T>();

    event_t mte2ToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
    event_t vToMte2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
    event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    float deSCaleVValue;
    if constexpr (isFp8) {
        deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
    }
    for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
        runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
        if (vec2S1Idx == vec2LoopLimit - 1) {
            runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
        }

        int64_t vec2CalcSize = runInfo.vec2S1RealSize * constInfo.dBasicBlock;
        // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
        int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64;
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        if (constInfo.dSizeV == dTemplateAlign64) {
            DataCopy(bmm2Ub, bmm2ResGm[runInfo.taskIdMod3][bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
        } else {
            DataCopyParams dataCopyParams;
            DataCopyPadParams dataCopyPadParams;
            dataCopyParams.blockCount = runInfo.vec2S1RealSize;
            dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
            dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
            dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
            DataCopyPad(bmm2Ub, bmm2ResGm[runInfo.taskIdMod3][bmm2SubBlockOffset + mm2ResInnerOffset],
                        dataCopyParams, dataCopyPadParams);
        }

        // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
        int64_t vec2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dBasicBlock;
        if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            DataCopy(vec2ResUb, this->vec2ResGm[runInfo.multiCoreIdxMod3][vec2SubBlockOffset + vec2ResInnerOffset], vec2CalcSize);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
            DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
        } else {
            int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
            float deSCalePreVValue = 1.0f;
            if constexpr (isFp8) {
                deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
            }
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, true>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            } else {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, true>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            }
        }

        if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock, deSCaleVValue);
            }
            GetDerived()->CopyOutAttentionOut(runInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
        } else if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            DataCopy(this->vec2ResGm[runInfo.multiCoreIdxMod3][vec2SubBlockOffset + vec2ResInnerOffset], vec2ResUb, vec2CalcSize);
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

S1S2_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::ProcessVec2(RunInfo<isInfer> &runInfo)
{
    if constexpr (bmm2Write2Ub) {
        ProcessVec2OnUb(runInfo);
    } else if constexpr (splitD) {
        ProcessVec2DSplit(runInfo);
    } else {
        // bmm2 result is on GM and global update data on UB
        runInfo.vec2S1BaseSize = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        LocalTensor<T> bmm2Ub = this->bmm2ResBuf[0].template Get<T>();

        event_t mte2ToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
        float deSCaleVValue;
        if constexpr (isFp8) {
            deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
        }
        for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
            runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
            if (vec2S1Idx == vec2LoopLimit - 1) {
                runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
            }

            int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfo.dSizeV == dTemplateAlign64) {
                DataCopy(bmm2Ub, bmm2ResGm[runInfo.taskIdMod3][bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = runInfo.vec2S1RealSize;
                dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
                DataCopyPad(bmm2Ub, bmm2ResGm[runInfo.taskIdMod3][bmm2SubBlockOffset + mm2ResInnerOffset],
                            dataCopyParams, dataCopyPadParams);
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            LocalTensor vec2ResInner = vec2ResUb[vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64];

            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
                DataCopy(vec2ResInner, bmm2Ub, vec2CalcSize);
            } else {
                int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                float deSCalePreVValue = 1.0f;
                if constexpr (isFp8) {
                    deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                }
                LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
                if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    } else {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    }
                } else {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    if (runInfo.s2LoopCount == runInfo.s2LoopStartIdx + 1) {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    } else {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    }   
                }
            }

            if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopStartIdx)) {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                        vec2ResInner, vec2ResInner, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue);
                }
                GetDerived()->CopyOutAttentionOut(runInfo, vec2ResInner, vec2S1Idx, vec2CalcSize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }
    return;
}

S1S2_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, RunInfo<isInfer> &runInfo)
{
    if constexpr (isInfer && hasAtten) {
        if (!constInfo.isRowInvalid || \
            this->attenMaskInfo.compressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return;
        }
        int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
        LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        bool isRowInvalidNeedUpdate = false;
        for (uint32_t i = 0; i < runInfo.vec2S1RealSize; i++) {
            float maxValue = maxTensor.GetValue(i);
            uint32_t checkValue = *(uint32_t*)&maxValue;
            if (checkValue == NEGATIVE_MIN_VAULE_FP32) {
                isRowInvalidNeedUpdate = true;
                break;
            }
        }
        if (isRowInvalidNeedUpdate) {
            RowInvalidUpdateVF<float, static_cast<uint32_t>(dVTemplateType)>(vec2ResUb, maxTensor, runInfo.vec2S1RealSize, constInfo.dSizeV);
        }
    }
}

S1S2_TEMPLATE
template <typename VEC2_RES_T>
__aicore__ inline void FlashAttentionScoreS1s2Const<S1S2_TEMPLATE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
                InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
                    dSizeAligned64, this->negativeFloatScalar, 0.0);
            }
        }
        RowInvalid(vec2ResUb, vec2S1Idx, runInfo);
        if constexpr (!POST_QUANT) {
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        } else {
            GetDerived()->PostQuant(runInfo, attenOut, vec2ResUb, vec2S1Idx);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    } else {
        if constexpr (!POST_QUANT) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
            attenOut = vec2ResUb;
        }
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    if constexpr (IsSameType<INPUT_T, float>::value) {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 3;
    } else {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    }
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
        if constexpr (isInfer) {
            if (constInfo.isGqa == 1) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            attenOutOffset = constInfo.n2GDv;
            if constexpr (isInfer) {
                if (constInfo.isGqa == 1) {
                    attenOutOffset = constInfo.dSizeV;
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            attenOutOffset = constInfo.bN2GDv;
        }
        if constexpr (isInfer) {
            if (constInfo.isBSNDOut == 1) {
                attenOutOffset = constInfo.n2GDv;
            }
        }
    }

    dataCopyParams.blockCount = runInfo.vec2S1RealSize;
    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
                attenOut, dataCopyParams);
}
#endif // FLASH_ATTENTION_SCORE_S1S2_CONST_H_
