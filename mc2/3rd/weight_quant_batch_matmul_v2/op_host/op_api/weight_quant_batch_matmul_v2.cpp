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

#include "weight_quant_batch_matmul_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(WeightQuantBatchMatmulV2);

constexpr int64_t TYPE_FP16 = 1;
constexpr int64_t TYPE_BF16 = 27;

const aclTensor* WeightQuantBatchMatmulV2(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional, bool transposeX, bool transposeWeight, int antiquantGroupSize, int64_t dtype,
    int innerPrecise, aclOpExecutor* executor)
{
    L0_DFX(
        WeightQuantBatchMatmulV2, x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional,
        quantOffsetOptional, biasOptional, transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise);
    DataType outType = x->GetDataType();
    if (dtype != -1) {
        outType = static_cast<DataType>(dtype);
    }
    auto output = executor->AllocTensor(outType, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        WeightQuantBatchMatmulV2,
        OP_INPUT(
            x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional),
        OP_OUTPUT(output), OP_ATTR(transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        WeightQuantBatchMatmulV2,
        OP_INPUT(
            x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional),
        OP_OUTPUT(output), OP_ATTR(transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return output;
}
} // namespace l0op