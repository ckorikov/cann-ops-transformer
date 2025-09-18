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
 * \file allto_allv_gmm_utils.h
 * \brief
 */
#ifndef __ALLTO_ALLV_GMM_UTILS_H__
#define __ALLTO_ALLV_GMM_UTILS_H__

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace ALLTO_ALLV_GMM {
using namespace AscendC;

constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;                          // 32: a block has 32 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2; // 2: a float16 data has two bytes

template <class AT_, class BT_, class CT_, class BiasT_, const MatmulConfig& MM_CFG = CFG_MDL>
struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

template <typename T>
__aicore__ inline T GreatestCommonDivisor(T a, T b)
{
    T c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename T>
__aicore__ inline T LeastCommonMultiple(T a, T b)
{
    return a * b / GreatestCommonDivisor(a, b);
}
} // namespace ALLTO_ALLV_GMM

#endif // __ALLTO_ALLV_GMM_UTILS_H__
