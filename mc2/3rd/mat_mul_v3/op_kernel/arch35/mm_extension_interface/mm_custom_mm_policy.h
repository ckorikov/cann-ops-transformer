/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file mm_custom_mm_policy.h
 * \brief
 */
#ifndef MM_CUSTOM_MM_POLICY_H
#define MM_CUSTOM_MM_POLICY_H

#include "lib/matmul_intf.h"
#include "mm_copy_cube_out.h"

namespace MatmulCommon {
template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class MMCustomMatmulPolicy : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE,
    BIAS_TYPE> {
public:
    using CopyCubeOut = MMCustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, McgShfMode::DUAL_DST_SPLIT_M>;
};
}  // namespace MatmulCommon
#endif  // MM_CUSTOM_MM_POLICY_H