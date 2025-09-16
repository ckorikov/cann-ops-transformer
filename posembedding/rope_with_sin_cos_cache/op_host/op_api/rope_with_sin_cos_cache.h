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
 * \file rope_with_sin_cos_cache.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_ROPE_WITH_SIN_COS_CACHE_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_ROPE_WITH_SIN_COS_CACHE_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> RopeWithSinCosCache(const aclTensor* positions, const aclTensor* queryIn,
                                                             const aclTensor* keyIn, const aclTensor* cosSinCache,
                                                             const aclIntArray *mropeSection, int64_t headSize,
                                                             bool isNeoxStyle, int64_t qStride, int64_t kStride,
                                                             int64_t numQHeads, int64_t numKHeads,
                                                             aclOpExecutor *executor);
}
#endif // OP_API_INC_LEVEL0_OP_ROPE_WITH_SIN_COS_CACHE_OP_H_
