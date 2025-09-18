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
 * \file moe_finalize_routing_common.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_COMMON
#define MOE_FINALIZE_ROUTING_COMMON

#include "kernel_operator.h"

namespace MoeFinalizeRouting {
using namespace AscendC;

constexpr int64_t ONE_BLK_SIZE = 32;
constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t PARALLEL_NUM = 2;

__aicore__ inline int64_t PadProcessInt32(int64_t param)
{
    return ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

__aicore__ inline int64_t Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}
} // namespace MoeFinalizeRouting
#endif // MOE_FINALIZE_ROUTING_COMMON
