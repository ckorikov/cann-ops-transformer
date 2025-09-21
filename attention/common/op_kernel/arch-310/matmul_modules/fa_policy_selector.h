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
 * \file fa_policy_selector.h
 * \brief
 */
#ifndef FA_POLICY_SELECTOR_H_
#define FA_POLICY_SELECTOR_H_
#include "fa_custom_matmul_policy.h"

// =================================L1 extension=============================
// =================================Bmm1 policy==============================
/* ------------------------------- S1 BaseSize 128 ------------------------------- */
namespace AscendC {
namespace Impl {
namespace Detail {
template<bool isFp32, DTemplateType dTemplateType, S2TemplateType s2TemplateType, S1TemplateType s1TemplateType, bool hasOptInput, bool isPa, bool hasRope>
struct Bmm1ConstPolicySelector {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

// bmm1 common
template<DTemplateType dTemplateType>
struct Bmm1ConstPolicySelectorCommonS1s2Align {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : std::conditional_t<
        dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128 ||
        dTemplateType == DTemplateType::Aligned192 || dTemplateType == DTemplateType::Aligned256,
        FACustomMatmulPolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>,
        MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>>{};
};

template<bool hasOptInput>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned192, S2TemplateType::Aligned128, S1TemplateType::Aligned128, hasOptInput, false, true> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2AlignRope<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned64, S2TemplateType::Aligned256, S1TemplateType::Aligned128, false, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2AlignDn<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<DTemplateType dTemplateType>
struct Bmm1ConstPolicySelector<false, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned128, false, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : std::conditional_t<
        dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128 ||
        dTemplateType == DTemplateType::Aligned192 || dTemplateType == DTemplateType::Aligned256,
        FACustomMatmulPolicyS1s2AlignDn<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>,
        MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>>{};
};

template<DTemplateType dTemplateType>
struct Bmm1ConstPolicySelector<false, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned128, true, false, false>
    : Bmm1ConstPolicySelectorCommonS1s2Align<dTemplateType> {};

template<DTemplateType dTemplateType, bool hasOptInput>
struct Bmm1ConstPolicySelector<true, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned128, hasOptInput, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : std::conditional_t<
        dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128,
        FACustomMatmulPolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>,
        MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>>{};
};

/* ------------------------------- S1 BaseSize 64 ------------------------------- */
template<DTemplateType dTemplateType, bool hasOptInput>
struct Bmm1ConstPolicySelector<false, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned64, hasOptInput, false, false>
    : Bmm1ConstPolicySelectorCommonS1s2Align<dTemplateType> {};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned64, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned128, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned192, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2DoubleBaseAlign<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned256, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2DoubleBaseAlign<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned64, S2TemplateType::Aligned256, S1TemplateType::Aligned64, true, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned128, S2TemplateType::Aligned256, S1TemplateType::Aligned64, true, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned192, S2TemplateType::Aligned256, S1TemplateType::Aligned64, true, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2DoubleBaseAlign<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm1ConstPolicySelector<false, DTemplateType::Aligned256, S2TemplateType::Aligned256, S1TemplateType::Aligned64, true, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyS1s2DoubleBaseAlign<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};
/* ========================================== Bmm2 Policy ========================================= */
/* ------------------------------- S1 Base128 S2 Split ------------------------------- */
template<bool isFp32, DTemplateType dTemplateType, S2TemplateType s2TemplateType, S1TemplateType s1TemplateType, bool isFp8, bool isPa>
struct Bmm2ConstPolicySelector {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

// bmm2 common
template<DTemplateType dTemplateType>
struct Bmm2ConstPolicySelectorCommonS1s2Align {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : std::conditional_t<
        dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128 ||
        dTemplateType == DTemplateType::Aligned192 || dTemplateType == DTemplateType::Aligned256,
        FACustomMatmul2PolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>,
        MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>>{};
};

template<>
struct Bmm2ConstPolicySelector<false, DTemplateType::Aligned64, S2TemplateType::Aligned256, S1TemplateType::Aligned128, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmul2PolicyS1s2AlignDn<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<DTemplateType dTemplateType>
struct Bmm2ConstPolicySelector<true, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned128, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : std::conditional_t<
        dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128,
        FACustomMatmul2PolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>,
        MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>>{};
};

template<DTemplateType dTemplateType>
struct Bmm2ConstPolicySelector<false, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned128, false, false>
    : Bmm2ConstPolicySelectorCommonS1s2Align<dTemplateType> {};

/* ------------------------------- S1 Base64 S2 Split ------------------------------- */
template<DTemplateType dTemplateType>
struct Bmm2ConstPolicySelector<true, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned64, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : std::conditional_t<
        dTemplateType == DTemplateType::Aligned64 || dTemplateType == DTemplateType::Aligned128,
        FACustomMatmul2PolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>,
        MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>>{};
};

template<DTemplateType dTemplateType>
struct Bmm2ConstPolicySelector<false, dTemplateType, S2TemplateType::Aligned128, S1TemplateType::Aligned64, false, false>
    : Bmm2ConstPolicySelectorCommonS1s2Align<dTemplateType> {};

template<>
struct Bmm2ConstPolicySelector<false, DTemplateType::Aligned64, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmul2PolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm2ConstPolicySelector<false, DTemplateType::Aligned128, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmul2PolicyS1s2Align<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm2ConstPolicySelector<false, DTemplateType::Aligned192, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmul2PolicyS1s2AlignDn<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Bmm2ConstPolicySelector<false, DTemplateType::Aligned256, S2TemplateType::Aligned256, S1TemplateType::Aligned64, false, false> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmul2PolicyS1s2AlignDn<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

/* ------------------------------- Infer Policy ------------------------------- */
template<bool isFp32, DTemplateType dTemplateType, S2TemplateType s2TemplateType, S1TemplateType s1TemplateType, bool hasOptInput, bool hasRope>
struct Bmm1ConstPolicySelector<isFp32, dTemplateType, s2TemplateType, s1TemplateType, hasOptInput, true, hasRope> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmulPolicyPa<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<bool isFp32, DTemplateType dTemplateType, S2TemplateType s2TemplateType, S1TemplateType s1TemplateType, bool isFp8>
struct Bmm2ConstPolicySelector<isFp32, dTemplateType, s2TemplateType, s1TemplateType, isFp8, true> {
    template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : FACustomMatmul2PolicyPa<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};
}
}
}

#endif //FA_POLICY_SELECTOR_H_
