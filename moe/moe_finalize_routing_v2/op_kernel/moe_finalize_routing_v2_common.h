/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_v2_common.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_COMMON
#define MOE_FINALIZE_ROUTING_V2_COMMON

#include "kernel_operator.h"

namespace MoeFinalizeRoutingV2 {
using namespace AscendC;

constexpr int64_t ONE_BLK_SIZE = 32;
constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t PARALLEL_NUM = 2;
constexpr int64_t INVALID_ROW_INDEX = -1;
constexpr int64_t MODE_VALUE_0 = 0;
constexpr int64_t MODE_VALUE_1 = 1;
constexpr int64_t MODE_VALUE_2 = 2;
constexpr int64_t MODE_VALUE_3 = 3;
constexpr int64_t BLOCK_BYTES = 32;

__aicore__ inline int64_t PadProcessInt32(int64_t param)
{
    return ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

__aicore__ inline int64_t Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(
    LocalTensor<T> inLocal, GlobalTensor<T> srcGm, DataCopyParams tokenCopyParams, DataCopyPadParams padParams)
{
#if __CCE_AICORE__ == 220
    DataCopyPad(inLocal, srcGm, tokenCopyParams, padParams);
#else
    int64_t elem =  tokenCopyParams.blockLen / sizeof(T);
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
__aicore__ inline void DataCopyPadExtCustom(
    LocalTensor<T> inLocal, GlobalTensor<T> srcGm, DataCopyExtParams tokenCopyParams, DataCopyPadExtParams<T> padParams)
{
#if __CCE_AICORE__ == 220
    DataCopyPad(inLocal, srcGm, tokenCopyParams, padParams);
#else
    int64_t elem =  tokenCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {tokenCopyParams.blockCount, static_cast<uint16_t>(alignElem / numPerBlock) , 0, 0};
        DataCopy(inLocal, srcGm, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock) , 0, 0};
        for (uint32_t i = 0; i < tokenCopyParams.blockCount; i++) {
            DataCopy(inLocal[i * alignElem], srcGm[i * elem], copyParams);
        }
    }
#endif
}

template <typename T, bool needBack = false, bool isAtomic = false>
__aicore__ inline void DataCopyCustom(GlobalTensor<T> dstGm, LocalTensor<T> inLocal, int64_t blockCount, int64_t blockLen)
{
    int64_t elem =  blockLen / sizeof(T);
    int64_t numPerBlock = sizeof(T) == 0 ? 1 : BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock) , 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
    } else {
        if (blockCount == 1) {
            if constexpr (needBack) {
                int64_t elemAlignDown = numPerBlock == 0 ? 0 :  elem / numPerBlock * numPerBlock;
                if (elemAlignDown != 0) {
                    DataCopyParams copyParams = {static_cast<uint16_t>(blockCount), static_cast<uint16_t>(elemAlignDown / numPerBlock) , 0, 0};
                    DataCopy(dstGm, inLocal, copyParams);
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

                    for (uint32_t i = 0; i < numPerBlock; i++) {
                        inLocal.SetValue(alignElem-1-i, inLocal.GetValue(elem - 1 - i));
                    }
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);

                    DataCopyParams copyParamslast = {1, 1, 0, 0};

                    DataCopy(dstGm[elem-numPerBlock], inLocal[elemAlignDown], copyParamslast);
                } else {
                    T tmp[BLOCK_BYTES];
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    for (uint32_t i = 0; i < elem; i++) {
                        tmp[i] = inLocal.GetValue(elem - 1 - i);
                    }
                    DataCopyParams copyParamslast = {1, 1, 0, 0};
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::S_MTE2);
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                    DataCopy(inLocal, dstGm[elem-numPerBlock], copyParamslast);
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    for (uint32_t i = 0; i < elem; i++) {
                        inLocal.SetValue(numPerBlock-1-i, tmp[i]);
                    }
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopy(dstGm[elem-numPerBlock], inLocal, copyParamslast);
                }

            } else if constexpr (isAtomic) {
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                for (uint32_t i = 0; i < alignElem - elem; i++) {
                    inLocal.SetValue(alignElem-1-i, T(0));
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);

                DataCopyParams copyParams = {static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock) , 0, 0};
                DataCopy(dstGm, inLocal, copyParams);
            } else {
                DataCopyParams copyParams = {static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock) , 0, 0};
                DataCopy(dstGm, inLocal, copyParams);
            }
        } else {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock) , 0, 0};
            for (uint32_t i = 0; i < blockCount; i++) {
                DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}
} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_V2_COMMON
