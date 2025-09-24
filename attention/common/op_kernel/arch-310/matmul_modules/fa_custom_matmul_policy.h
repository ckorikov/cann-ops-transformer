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
 * \file fa_custom_matmul_policy.h
 * \brief
 */
#ifndef FA_CUSTOM_MATMUL_POLICY_H_
#define FA_CUSTOM_MATMUL_POLICY_H_

#include "copy_cube_in/copy_cube_in_s1s2_align/fa_copy_cube_in_left_s1s2_align.h"
#include "copy_cube_in/copy_cube_in_s1s2_align/fa_copy_cube_in_right_s1s2_align.h"
#include "copy_cube_in/copy_cube_in_s1s2_align/fa_copy_cube_in_left_s1s2_align_rope.h"
#include "copy_cube_in/copy_cube_in_s1s2_align/fa_copy_cube_in_right_s1s2_align_rope.h"
#include "copy_cube_in/copy_cube_in_s1s2_align/fa_copy_cube_in_right_s1s2_doublebase_align.h"
#include "copy_cube_in/copy_cube_in_s1s2_align_dn/fa_copy_cube_in_left_s1s2_align_dn.h"
#include "copy_cube_in/copy_cube_in_s1s2_align_dn/fa_copy_cube_in_right_s1s2_align_dn.h"
#include "copy_cube_in/copy_cube_in_s1s2_align_dn/fa_copy_cube2_in_right_s1s2_align_dn.h"
#include "copy_cube_in/copy_cube_in_pa/fa_copy_cube_in_left_pa.h"
#include "copy_cube_in/copy_cube_in_pa/fa_copy_cube_in_right_pa.h"
#include "copy_cube_in/copy_cube_in_pa/fa_copy_cube2_in_right_pa.h"
#include "cube_in_buffer/fa_cube_in_buffer.h"
#include "cube_in_buffer/fa_cube_in_buffer_double_base.h"
#include "lib/../impl/matmul/policy/matmul_policy.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmulPolicyS1s2Align : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FAFlagData;
    using CubeInBufferA = FACubeInBufferGeneral<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = FACopyCubeInLeftS1s2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FACubeInBufferGeneral<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FACopyCubeInRightS1s2Align<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmulPolicyS1s2DoubleBaseAlign : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FAFlagData;
    using CubeInBufferA = FACubeInBufferGeneral<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = FACopyCubeInLeftS1s2Align<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FACubeInBufferDoubleBase<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FACopyCubeInRightS1s2DoubleBaseAlign<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmulPolicyS1s2AlignDn : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FAFlagData;
    using CubeInBufferA = FACubeInBufferGeneral<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = FACopyCubeInLeftS1s2AlignDn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FACubeInBufferGeneral<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FACopyCubeInRightS1s2AlignDn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeOut = CopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, McgShfMode::DUAL_DST_SPLIT_N>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmul2PolicyS1s2Align : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FAFlagData;
    using CubeInBufferB = FACubeInBufferGeneral<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB =
        FACopyCubeInRightS1s2Align<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmul2PolicyS1s2AlignDn : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FAFlagData;
    using CubeInBufferB = FACubeInBufferDoubleBase<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB =
        FACopyCube2InRightS1s2AlignDn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmulPolicyPa : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FaPaFlagData;
    using CubeInBufferA = FACubeInBufferGeneral<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T> , MM_CFG>;
    using CubeInBufferB = FACubeInBufferGeneral<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T> , MM_CFG>;
    using CopyCubeInA = FaCopyCubeInLeftPa<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FaCopyCubeInRightPa<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmul2PolicyPa : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FaPaFlagData;
    using CubeInBufferB = FACubeInBufferGeneral<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T> , MM_CFG>;
    using CopyCubeInB = FaCopyCube2InRightPa<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
};

template<const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FACustomMatmulPolicyS1s2AlignRope : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FAFlagDataWhole<true>;
    using CubeInBufferA = FACubeInBufferGeneral<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInA = FACopyCubeInLeftS1s2AlignRope<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FACubeInBufferGeneral<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = FACopyCubeInRightS1s2AlignRope<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
};
}
}
}

#endif // FA_CUSTOM_MATMUL_POLICY_H_
