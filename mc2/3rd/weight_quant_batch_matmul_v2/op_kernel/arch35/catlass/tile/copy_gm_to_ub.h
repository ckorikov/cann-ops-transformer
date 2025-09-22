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
#ifndef ARCH35_CATLASS_TILE_COPY_GM_TO_UB_H
#define ARCH35_CATLASS_TILE_COPY_GM_TO_UB_H

#include "../utils/device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

template <typename T>
DEVICE void CopyGmToUbIntervalDataCopy(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t blockCount, uint32_t blockLen,
    uint32_t dstInnerLength, uint32_t srcInnerLength)
{
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(dstInnerLength >= blockLen, {
        X_LOG("dstInnerLength[%d] should be larger than blockLen[%d].", dstInnerLength, blockLen);
    });
#endif
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    X_LOG(
        "DataCopyPad2D blockCount %d blockLen %d srcStride %d dstStride %d", params.blockCount, params.blockLen,
        params.srcStride, params.dstStride);
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(params.blockLen > 0, { X_LOG("blockLen[%d] should be larger than 0.", params.blockLen); });
#endif
    DataCopyPad(dst, src, params, padParams);
}

} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif