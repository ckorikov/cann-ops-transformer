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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#include "kernel_tensor.h"
namespace fa {
using AscendC::LocalTensor;
#ifndef __CCE_KT_TEST__
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false


template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, float keepProb)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();

    constexpr uint32_t blockStride = ubN >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __VEC_SCOPE__{
        RegTensor<float> vreg_x_sum_even;
        RegTensor<float> vreg_x_sum_odd;
        RegTensor<float> vreg_x_sum_1_even;
        RegTensor<float> vreg_x_sum_1_odd;
        RegTensor<float> vreg_x_sum0;
        RegTensor<float> vreg_x_sum1;
        RegTensor<half> vreg_x_exp_even_f16;
        RegTensor<half> vreg_x_exp_odd_f16;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;

        RegTensor<float> vreg_x_exp_even;
        RegTensor<float> vreg_x_exp_odd;
        RegTensor<float> vreg_x_f32_a;
        RegTensor<float> vreg_x_f32_b;
        RegTensor<float> vreg_x_exp_even_1;
        RegTensor<float> vreg_x_exp_odd_1;
        RegTensor<half> vreg_x_exp_even_f16_1;
        RegTensor<half> vreg_x_exp_odd_f16_1;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

        RegTensor<float> vreg_x_f32_1_a;
        RegTensor<float> vreg_x_f32_1_b;
        RegTensor<half> vreg_x_exp_f16_pack;
        RegTensor<half> vreg_x_exp_f16_1_pack;
        RegTensor<half> vreg_x_exp_f16_packa;
        RegTensor<half> vreg_x_exp_f16_1_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
        MaskReg preg_100;
        MaskReg preg_101;
        MaskReg preg_134;
        MaskReg preg_135;
        MaskReg preg_136;
        preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
        preg_135 = CreateMask<T, MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)128ULL;
        preg_136 = UpdateMask<uint16_t>(sreg_92);
        MaskReg preg_108;
        RegTensor<float> src_00a, src_01a, src_02a, src_03a;
        RegTensor<float> src_00b, src_01b, src_02b, src_03b;
        RegTensor<float> src_10a, src_11a, src_12a, src_13a;
        RegTensor<float> src_10b, src_11b, src_12b, src_13b;
        RegTensor<float> max_0a, max_1a, max_2a, max_3a;
        RegTensor<float> max_0b, max_1b, max_2b, max_3b;
        RegTensor<float> vreg_min;

        __ubuf__ float *src0_ub = (__ubuf__ float*) input_x_local_UB;
        __ubuf__ float *src0_ub_1 = src0_ub + m;
        __ubuf__ T2 *x_exp_1 = x_exp + (ubN * 4);
        __ubuf__ float *src0_ub1 = src0_ub + m * 2;
        __ubuf__ float *src0_ub1_1 = src0_ub + m * 3;
        __ubuf__ float *src0_ub2 = src0_ub + m * 4;
        __ubuf__ float *src0_ub2_1 = src0_ub + m * 5;
        __ubuf__ float *src0_ub3 = src0_ub + m * 6;
        __ubuf__ float *src0_ub3_1 = src0_ub + m * 7;

        Duplicate(max_0a, minValue);
        Duplicate(max_0b, minValue);
        Duplicate(max_1a, minValue);
        Duplicate(max_1b, minValue);
        Duplicate(max_2a, minValue);
        Duplicate(max_2b, minValue);
        Duplicate(max_3a, minValue);
        Duplicate(max_3b, minValue);
        Duplicate(vreg_min, minValue);
        for (uint16_t i = originN; i < ubN; ++i) {
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
        }
        mem_bar(VST_VLD);

        preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            DataCopy(src_00a, src0_ub + iter_m * m * 8);
            Max(max_0a, max_0a, src_00a, preg_108);
            DataCopy(src_00b, src0_ub_1 + iter_m * m * 8);
            Max(max_0b, max_0b, src_00b, preg_108);
            DataCopy(src_01a, src0_ub1 + iter_m * m * 8);
            Max(max_1a, max_1a, src_01a, preg_108);
            DataCopy(src_01b, src0_ub1_1 + iter_m * m * 8);
            Max(max_1b, max_1b, src_01b, preg_108);
            DataCopy(src_02a, src0_ub2 + iter_m * m * 8);
            Max(max_2a, max_2a, src_02a, preg_108);
            DataCopy(src_02b, src0_ub2_1 + iter_m * m * 8);
            Max(max_2b, max_2b, src_02b, preg_108);
            DataCopy(src_03a, src0_ub3 + iter_m * m * 8);
            Max(max_3a, max_3a, src_03a, preg_108);
            DataCopy(src_03b, src0_ub3_1 + iter_m * m * 8);
            Max(max_3b, max_3b, src_03b, preg_108);
        }

        Max(max_0a, max_0a, max_1a, preg_108);
        Max(max_0b, max_0b, max_1b, preg_108);
        Max(max_2a, max_2a, max_3a, preg_108);
        Max(max_2b, max_2b, max_3b, preg_108);
        Max(max_0a, max_0a, max_2a, preg_108);
        Max(max_0b, max_0b, max_2b, preg_108);
        Max(max_0a, max_0a, max_0b, preg_108);
        Muls(max_0a, max_0a, scale, preg_108);

        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)new_global_max, max_0a, preg_108);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_odd, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1_odd, 0, preg_134);
        for (uint16_t i0 = 0; i0 < uint16_t(ubN / 4); ++i0) {
            DataCopy(vreg_x_f32_a, input_x_local_UB + i0 * m);
            DataCopy(vreg_x_f32_b, input_x_local_UB + ubN * m / 2 + i0 * m);
            DataCopy(vreg_x_f32_1_a, input_x_local_UB + ubN * m / 4 + i0 * m);
            DataCopy(vreg_x_f32_1_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);

            Muls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            Muls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);

            FusedExpSub(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134);
            FusedExpSub(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134);

            Muls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            Muls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);

            FusedExpSub(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134);
            FusedExpSub(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134);
            if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_odd_1, preg_135);
                DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
                /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            } else {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_135);
                DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
                /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);    
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);    
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            }
        }
        Add(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        Add(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
    }
}

template <typename T, typename T2, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, float keepProb)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();

    constexpr uint32_t blockStride = ubN >> 1 | 0x1;
    constexpr uint32_t repeatStride = 1;
    __VEC_SCOPE__{
        RegTensor<float> vreg_x_sum_even;
        RegTensor<float> vreg_x_sum_odd;
        RegTensor<float> vreg_x_sum_1_even;
        RegTensor<float> vreg_x_sum_1_odd;
        RegTensor<float> vreg_x_sum0;
        RegTensor<float> vreg_x_sum1;
        RegTensor<half> vreg_x_exp_even_f16;
        RegTensor<half> vreg_x_exp_odd_f16;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;

        RegTensor<float> vreg_x_exp_even;
        RegTensor<float> vreg_x_exp_odd;
        RegTensor<float> vreg_x_f32_a;
        RegTensor<float> vreg_x_f32_b;
        RegTensor<float> vreg_x_exp_even_1;
        RegTensor<float> vreg_x_exp_odd_1;
        RegTensor<half> vreg_x_exp_even_f16_1;
        RegTensor<half> vreg_x_exp_odd_f16_1;
        RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
        RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

        RegTensor<float> vreg_x_f32_1_a;
        RegTensor<float> vreg_x_f32_1_b;
        RegTensor<float> vreg_x_max_f32_b;
        RegTensor<half> vreg_x_exp_f16_pack;
        RegTensor<half> vreg_x_exp_f16_1_pack;
        RegTensor<half> vreg_x_exp_f16_packa;
        RegTensor<half> vreg_x_exp_f16_1_packa;

        RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
        RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

        MaskReg preg_100;
        MaskReg preg_101;
        MaskReg preg_134;
        MaskReg preg_135;
        MaskReg preg_136;
        preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
        preg_135 = CreateMask<T, MaskPattern::ALL>();
        uint32_t sreg_92 = (uint32_t)128ULL;
        preg_136 = UpdateMask<uint16_t>(sreg_92);
        MaskReg preg_108;
        RegTensor<float> src_00a, src_01a, src_02a, src_03a;
        RegTensor<float> src_00b, src_01b, src_02b, src_03b;
        RegTensor<float> src_10a, src_11a, src_12a, src_13a;
        RegTensor<float> src_10b, src_11b, src_12b, src_13b;
        RegTensor<float> max_0a, max_1a, max_2a, max_3a;
        RegTensor<float> max_0b, max_1b, max_2b, max_3b;
        RegTensor<float> vreg_min;

        __ubuf__ float *src0_ub = (__ubuf__ float*) input_x_local_UB;
        __ubuf__ float *src0_ub_1 = (__ubuf__ float*) input_x_local_UB + m;
        __ubuf__ T2 *x_exp_1 = x_exp + (ubN * 4);
        __ubuf__ float *src0_ub1 = src0_ub + m * 2;
        __ubuf__ float *src0_ub1_1 = src0_ub + m * 3;
        __ubuf__ float *src0_ub2 = src0_ub + m * 4;
        __ubuf__ float *src0_ub2_1 = src0_ub + m * 5;
        __ubuf__ float *src0_ub3 = src0_ub + m * 6;
        __ubuf__ float *src0_ub3_1 = src0_ub + m * 7;

        Duplicate(max_0a, minValue);
        Duplicate(max_0b, minValue);
        Duplicate(max_1a, minValue);
        Duplicate(max_1b, minValue);
        Duplicate(max_2a, minValue);
        Duplicate(max_2b, minValue);
        Duplicate(max_3a, minValue);
        Duplicate(max_3b, minValue);
        Duplicate(vreg_min, minValue);
        for (uint16_t i = originN; i < ubN; ++i) {
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
        }
        mem_bar(VST_VLD);

        preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 8); ++iter_m) {
            DataCopy(src_00a, src0_ub + iter_m * m * 8);
            Max(max_0a, max_0a, src_00a, preg_108);
            DataCopy(src_00b, src0_ub_1 + iter_m * m * 8);
            Max(max_0b, max_0b, src_00b, preg_108);
            DataCopy(src_01a, src0_ub1 + iter_m * m * 8);
            Max(max_1a, max_1a, src_01a, preg_108);
            DataCopy(src_01b, src0_ub1_1 + iter_m * m * 8);
            Max(max_1b, max_1b, src_01b, preg_108);
            DataCopy(src_02a, src0_ub2 + iter_m * m * 8);
            Max(max_2a, max_2a, src_02a, preg_108);
            DataCopy(src_02b, src0_ub2_1 + iter_m * m * 8);
            Max(max_2b, max_2b, src_02b, preg_108);
            DataCopy(src_03a, src0_ub3 + iter_m * m * 8);
            Max(max_3a, max_3a, src_03a, preg_108);
            DataCopy(src_03b, src0_ub3_1 + iter_m * m * 8);
            Max(max_3b, max_3b, src_03b, preg_108);
        }
        DataCopy(vreg_x_max_f32_b, new_global_max);
        Max(max_0a, max_0a, max_1a, preg_108);
        Max(max_0b, max_0b, max_1b, preg_108);
        Max(max_2a, max_2a, max_3a, preg_108);
        Max(max_2b, max_2b, max_3b, preg_108);
        Max(max_0a, max_0a, max_2a, preg_108);
        Max(max_0b, max_0b, max_2b, preg_108);
        Max(max_0a, max_0a, max_0b, preg_108);
        Muls(max_0a, max_0a, scale, preg_108);
        Max(max_0a, max_0a, vreg_x_max_f32_b, preg_108);

        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max_0a, preg_134);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)new_global_max, max_0a, preg_108);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);    

        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_odd, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1_even, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1_odd, 0, preg_134);
        for(uint16_t i0 = 0; i0 < uint16_t(ubN / 4); ++i0) {
            DataCopy(vreg_x_f32_a, input_x_local_UB + i0 * m);
            DataCopy(vreg_x_f32_b, input_x_local_UB + ubN * m / 2 + i0 * m);
            DataCopy(vreg_x_f32_1_a, input_x_local_UB + ubN * m / 4 + i0 * m);
            DataCopy(vreg_x_f32_1_b, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);

            Muls(vreg_x_f32_a, vreg_x_f32_a, scale, preg_108);
            Muls(vreg_x_f32_b, vreg_x_f32_b, scale, preg_108);

            FusedExpSub(vreg_x_exp_even, vreg_x_f32_a, max_0a, preg_134);
            FusedExpSub(vreg_x_exp_odd, vreg_x_f32_b, max_0a, preg_134);

            Muls(vreg_x_f32_1_a, vreg_x_f32_1_a, scale, preg_108);
            Muls(vreg_x_f32_1_b, vreg_x_f32_1_b, scale, preg_108);

            FusedExpSub(vreg_x_exp_even_1, vreg_x_f32_1_a, max_0a, preg_134);
            FusedExpSub(vreg_x_exp_odd_1, vreg_x_f32_1_b, max_0a, preg_134);

            if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_odd_1, preg_135);
                DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
                /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);     
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136); 
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);

            } else {
                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_even, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_odd, preg_135);
                DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

                Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_even_1, preg_135);
                Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_odd_1, preg_135);

                DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

                /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                  个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136); 
                Add(vreg_x_sum_even, vreg_x_exp_even, vreg_x_sum_even, preg_134);
                Add(vreg_x_sum_odd, vreg_x_exp_odd, vreg_x_sum_odd, preg_134);

                DataCopy<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136); 
                Add(vreg_x_sum_1_even, vreg_x_exp_even_1, vreg_x_sum_1_even, preg_134);
                Add(vreg_x_sum_1_odd, vreg_x_exp_odd_1, vreg_x_sum_1_odd, preg_134);
            }
        }
        Add(vreg_x_sum0, vreg_x_sum_odd, vreg_x_sum_even, preg_134);
        Add(vreg_x_sum1, vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
        RegTensor<float> vreg_l0;
        DataCopy(vreg_l0, new_global_sum);
        Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
        Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
        DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134); 
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor,
                                       const uint32_t m, const uint32_t n, const uint32_t originN,
                                       const T scale, const T minValue, float keepProb)
{
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdate<T, T2, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            m, n, originN, scale, minValue, keepProb);
    } else {
        ProcessVec1DnUpdate<T, T2, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            m, n, originN, scale, minValue, keepProb);
    }
}

template <typename T>
__aicore__ inline void BroadcastMaxSum(const LocalTensor<T>& outTensor, const LocalTensor<T> &oriTensor,
                                       uint32_t vecS1RealSize)
{
    __ubuf__ float *out_ub = (__ubuf__ T*)outTensor.GetPhyAddr();
    __ubuf__ float *ori_ub = (__ubuf__ T*)oriTensor.GetPhyAddr();

    // Align8, broadcast one element to 8 elements, one register can store 64 elements,
    // so we can handle 64 / 8 = 8 elements per loop.
    uint16_t loopM = (vecS1RealSize + 7) >> 3;
    __VEC_SCOPE__{
        RegTensor<float> broadcast_reg;
        MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopM; ++i) {
            DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(
                broadcast_reg, ori_ub + i * 8);
            DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)out_ub + i * 64, broadcast_reg, preg_all); 
        }
    }
}
#else
template <typename T, typename T2, bool isUpdate = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor,
                                       const uint32_t m, const uint32_t n, const uint32_t originN,
                                       const T scale, const T minValue, float keepProb)
{
}

template <typename T>
__aicore__ inline void BroadcastMaxSum(const LocalTensor<T>& outTensor, const LocalTensor<T> &oriTensor,
                                       uint32_t vecS1RealSize)
{
}
#endif
}
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_