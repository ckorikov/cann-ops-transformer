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
 * \file test_moe_token_unpermute_with_ep.h
 * \brief
 */
#ifndef _MOE_TOKEN_UNPERMUTE_WITH_EP_TILING_H_
#define _MOE_TOKEN_UNPERMUTE_WITH_EP_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct MoeTokenUnpermuteWithEpTilingData {
    int64_t hidden_size = 0;
    int64_t permuted_probs_grad_length = 0;
    int64_t top_k = 0;
    int64_t start = 0;
    int64_t end = 0;
    int64_t num_out_tokens = 0;
    int64_t hidden_splited_length = 0;
    int64_t hidden_splited_num = 0;
    int64_t hidden_splited_remain = 0;
    int64_t tokens_core_length = 0;
    int64_t tokens_core_remain = 0;
    int64_t tokens_splited_length = 0;
    int64_t tokens_splited_num = 0;
    int64_t tokens_splited_remain = 0;
    int64_t buffer_num = 0;
};

inline void InitMoeTokenUnpermuteWithEpTilingData(uint8_t* tiling, MoeTokenUnpermuteWithEpTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeTokenUnpermuteWithEpTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    MoeTokenUnpermuteWithEpTilingData tilingData;  \
    InitMoeTokenUnpermuteWithEpTilingData(tilingPointer, &tilingData)
#endif