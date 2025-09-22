/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_WEIGHT_QUANT_BATCH_MATMUL_V2_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_WEIGHT_QUANT_BATCH_MATMUL_V2_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor* WeightQuantBatchMatmulV2(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional, bool transposeX, bool transposeWeight, int antiquantGroupSize, int64_t dtype,
    int innerPrecise, aclOpExecutor* executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_WEIGHT_QUANT_BATCH_MATMUL_V2_H