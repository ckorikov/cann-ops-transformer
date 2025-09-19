/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

#include "aclnn_rotary_position_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor* x, const aclTensor* cos,
                                                                     const aclTensor* sin, int64_t mode, aclTensor* out,
                                                                     uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerRotaryPositionEmbedding(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream);

aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                         int64_t mode, aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor)
{
    return aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize(x, cos, sin, mode, out, workspaceSize, executor);
}

aclnnStatus aclnnRotaryPositionEmbedding(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    return aclnnInnerRotaryPositionEmbedding(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
