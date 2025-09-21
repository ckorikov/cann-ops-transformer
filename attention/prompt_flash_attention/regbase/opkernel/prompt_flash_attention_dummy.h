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

/*!
 * \file prompt_flash_attention_dummy.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_DUMMY_H
#define PROMPT_FLASH_ATTENTION_DUMMY_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

template<typename T>
class PromptFlashAttentionDummy {
public:
    __aicore__ inline PromptFlashAttentionDummy() {};
    __aicore__ inline void Init(__gm__ uint8_t* attentionOut, const FlashAttentionScoreSimplifiedTilingData* __restrict tiling);
    __aicore__ inline void Process();

protected:
    const FlashAttentionScoreSimplifiedTilingData* __restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template<typename T>
__aicore__ inline void PromptFlashAttentionDummy<T>::Init(__gm__ uint8_t *attentionOut,
    const FlashAttentionScoreSimplifiedTilingData* __restrict tiling) {
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void PromptFlashAttentionDummy<T>::Process() {
    uint32_t blockIdx = GetBlockIdx();
}
#endif  // PROMPT_FLASH_ATTENTION_DUMMY_H