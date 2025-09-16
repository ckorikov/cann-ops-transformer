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
 * \file rope_with_sin_cos_cache.cpp
 * \brief
 */

#include "rope_with_sin_cos_cache.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RopeWithSinCosCache);

// AICORE算子kernel
const std::tuple<aclTensor*, aclTensor*> RoepWithSinCosCacheAICore(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclIntArray* mropeSection, int64_t headSize, bool isNeoxStyle, int64_t qStride, int64_t kStride,
    int64_t numQHeads, int64_t numKHeads, aclTensor* queryOut, aclTensor* keyOut, aclOpExecutor* executor)
{
    L0_DFX(
        RoepWithSinCosCacheAICore, positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle, qStride,
        kStride, numQHeads, numKHeads, queryOut, keyOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        RopeWithSinCosCache, OP_INPUT(positions, queryIn, keyIn, cosSinCache), OP_OUTPUT(queryOut, keyOut),
        OP_ATTR(numQHeads, numKHeads, headSize, mropeSection, qStride, kStride, isNeoxStyle));
    return std::tie(queryOut, keyOut);
}

const std::tuple<aclTensor*, aclTensor*> RopeWithSinCosCache(
    const aclTensor* positions, const aclTensor* queryIn, const aclTensor* keyIn, const aclTensor* cosSinCache,
    const aclIntArray* mropeSection, int64_t headSize, bool isNeoxStyle, int64_t qStride, int64_t kStride,
    int64_t numQHeads, int64_t numKHeads, aclOpExecutor* executor)
{
    // 根据推导出的输出shape申请输出tensor
    auto queryOut = executor->AllocTensor(queryIn->GetViewShape(), queryIn->GetDataType(), queryIn->GetViewFormat());
    auto keyOut = executor->AllocTensor(keyIn->GetViewShape(), keyIn->GetDataType(), keyIn->GetViewFormat());

    return RoepWithSinCosCacheAICore(
        positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle, qStride, kStride, numQHeads,
        numKHeads, queryOut, keyOut, executor);
}
} // namespace l0op