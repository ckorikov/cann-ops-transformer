/*	
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
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::array<aclTensor*, 2> BNTrainingUpdateGrad(const aclTensor* gradOut, const aclTensor* x,
                                                     const aclTensor* saveMean, const aclTensor* saveInvstd, float eps,
                                                     aclOpExecutor* executor);
const std::array<aclTensor*, 2> BN3DTrainingUpdateGrad(const aclTensor* gradOut, const aclTensor* x,
                                                       const aclTensor* saveMean, const aclTensor* saveInvstd,
                                                       float eps, aclOpExecutor* executor);

const aclTensor* BNTrainingReduceGrad(const aclTensor* gradOut, const aclTensor* x, const aclTensor* gradWeight,
                                      const aclTensor* gradBias, const aclTensor* weight, const aclTensor* saveMean,
                                      const aclTensor* saveInvstd, float eps, aclOpExecutor* executor);
const aclTensor* BN3DTrainingReduceGrad(const aclTensor* gradOut, const aclTensor* x, const aclTensor* gradWeight,
                                        const aclTensor* gradBias, const aclTensor* weight, const aclTensor* saveMean,
                                        const aclTensor* saveInvstd, float eps, aclOpExecutor* executor);

const aclTensor* BNInferGrad(const aclTensor* gradOut, const aclTensor* weight, const aclTensor* runningVar, float eps,
                             aclOpExecutor* executor);

constexpr size_t BN_GRAD_V3_OUTPUT_NUM = 3;
const std::array<aclTensor*, BN_GRAD_V3_OUTPUT_NUM> BatchNormGradV3(const aclTensor* gradOut,
                                                                    const aclTensor* input,
                                                                    const aclTensor* weight,
                                                                    const aclTensor* runningMean,
                                                                    const aclTensor* runningVar,
                                                                    const aclTensor* saveMean,
                                                                    const aclTensor* saveInvstd,
                                                                    bool training, float eps,
                                                                    aclOpExecutor* executor);
}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_GRAD_OP_H_
