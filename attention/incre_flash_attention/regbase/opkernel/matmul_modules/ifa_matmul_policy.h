/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file ifa_matmul_policy.h
 * \brief
 */
#ifndef IFA_MATMUL_POLICY_H
#define IFA_MATMUL_POLICY_H

#include "ifa_flag_data.h"
#include "ifa_cube_in_buffer.h"
#include "ifa_copy_cube_in.h"
#include "lib/../impl/matmul/policy/matmul_policy.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE, const auto& MM_CFG, typename MM_CB>
class IFAMatmulPolicyNormal : public MatmulPolicy<MM_CFG,IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
 public:
  using UserDefDataType = IFAFlagData;
  using CubeInBufferA = AscendC::Impl::Detail::IFACubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
  using CopyCubeInA = AscendC::Impl::Detail::IFACopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif  // IFA_MATMUL_POLICY_H
