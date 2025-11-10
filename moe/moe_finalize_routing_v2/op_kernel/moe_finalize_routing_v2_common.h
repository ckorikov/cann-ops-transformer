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

__aicore__ inline int64_t PadProcessInt32(int64_t param)
{
    return ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

__aicore__ inline int64_t Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}

template <typename T, typename CopyParamType, typename PadParamType>
__aicore__ inline void DataCopyPadCustom(
    LocalTensor<T> inLocal, GlobalTensor<T> srcGm, CopyParamType padCopyParams, PadParamType padParams)
{
#if __CCE_AICORE__ == 200
    int64_t elem = padCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);
    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {padCopyParams.blockCount, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(inLocal, srcGm, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        for (uint32_t i = 0; i < padCopyParams.blockCount; i++) {
            DataCopy(inLocal[i * alignElem], srcGm[i * elem], copyParams);
        }
    }
#else
    DataCopyPad(inLocal, srcGm, padCopyParams, padParams);
#endif
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(
    GlobalTensor<T> dstGm, LocalTensor<T> outLocal, DataCopyParams padCopyParams)
{
#if __CCE_AICORE__ == 200
    int64_t elem = padCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {padCopyParams.blockCount, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, outLocal, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        for (uint32_t i = 0; i < padCopyParams.blockCount; i++) {
            DataCopy(dstGm[i * elem], outLocal[i * alignElem], copyParams);
        }
    }
#else
    DataCopyPad(dstGm, outLocal, padCopyParams);
#endif
}

} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_V2_COMMON
