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
 * \file moe_token_unpermute_common.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_COMMON_H
#define MOE_TOKEN_UNPERMUTE_COMMON_H

#include "kernel_operator.h"

namespace MoeTokenUnPermute {
using namespace AscendC;

constexpr int64_t BLOCK_BYTES = 32;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(
    LocalTensor<T> inLocal, GlobalTensor<T> srcGm, DataCopyExtParams tokenCopyParams, DataCopyPadExtParams<T> padParams)
{
#if __CCE_AICORE__ == 220
    DataCopyPad(inLocal, srcGm, tokenCopyParams, padParams);
#else
    int64_t elem = tokenCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {tokenCopyParams.blockCount, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(inLocal, srcGm, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        for (uint32_t i = 0; i < tokenCopyParams.blockCount; i++) {
            DataCopy(inLocal[i * alignElem], srcGm[i * elem], copyParams);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    GlobalTensor<T> dstGm, LocalTensor<T> inLocal, int64_t blockCount, int64_t blockLen)
{
    int64_t elem = blockLen / sizeof(T);
    int64_t numPerBlock = sizeof(T) == 0 ? 1 : BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {
            static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
    } else {
        if (blockCount == 1) {
            DataCopyParams copyParams = {
                static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
            DataCopy(dstGm, inLocal, copyParams);
        }
        else {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
            for (uint32_t i = 0; i < blockCount; i++) {
                DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
                pipe_barrier(PIPE_MTE3);
            }
        }
    }
}

} // namespace MoeTokenUnPermute
#endif // MOE_TOKEN_UNPERMUTE_COMMON_H
