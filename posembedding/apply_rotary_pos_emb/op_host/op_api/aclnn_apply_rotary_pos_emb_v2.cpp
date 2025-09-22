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

#include "aclnn_apply_rotary_pos_emb_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerApplyRotaryPosEmbGetWorkspaceSize(aclTensor* queryRef, aclTensor* keyRef,
                                                               const aclTensor* cos, const aclTensor* sin,
                                                               int64_t layout, char* rotaryMode,
                                                               uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerApplyRotaryPosEmb(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(aclTensor* queryRef, aclTensor* keyRef, const aclTensor* cos,
                                                     const aclTensor* sin, int64_t layout,
                                                     char* rotaryMode, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    aclnnStatus ret = aclnnInnerApplyRotaryPosEmbGetWorkspaceSize(queryRef, keyRef, cos, sin, layout,
                                                                  rotaryMode, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnApplyRotaryPosEmbV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerApplyRotaryPosEmb(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
