/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_token_unpermute_with_ep.cpp
 * \brief
 */
#include "moe_token_unpermute_with_ep.h"
#include "kernel_operator.h"

#if !defined(DTYPE_PERMUTED_TOKENS)
#define DTYPE_PERMUTED_TOKENS bfloat16_t
#endif
#if !defined(DTYPE_PROBS)
#define DTYPE_PROBS DTYPE_PERMUTED_TOKENS
#endif

#define MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(T1, T2, T3, haveProbs, isUnpermute)                      \
    do {                                                                                          \
        KernelMoeTokenUnpermuteWithEp<T1, T2, T3, haveProbs, isUnpermute> op;                     \
        op.Init(permuted_tokens, sorted_indices, probs, unpermuted_tokens, tiling_data, &t_pipe); \
        op.Process();                                                                             \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_unpermute_with_ep(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data_in, tiling);
    const MoeTokenUnpermuteWithEpTilingData* __restrict tiling_data = &tiling_data_in;
    TPipe t_pipe;
    if (TILING_KEY_IS(0)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, DTYPE_PROBS, false, true);
    } else if (TILING_KEY_IS(1)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, float, true, true);
    } else if (TILING_KEY_IS(2)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, half, true, true);
    } else if (TILING_KEY_IS(3)) {
        MOE_TOKEN_UNPERMUTE_WITH_EP_IMPL(DTYPE_PERMUTED_TOKENS, int32_t, bfloat16_t, true, true);
    }
}
