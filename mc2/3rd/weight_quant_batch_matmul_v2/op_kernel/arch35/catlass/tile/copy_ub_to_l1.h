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
#ifndef ARCH35_CATLASS_TILE_COPY_UB_TO_L1_H
#define ARCH35_CATLASS_TILE_COPY_UB_TO_L1_H

#include "../utils/device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
using AscendC::DataCopyParams;
using AscendC::LocalTensor;

template <typename T>
DEVICE void CopyUbToL1IntervalDataCopy(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t blockCount, uint32_t blockLen,
    uint32_t dstInnerLength, uint32_t srcInnerLength)
{
    DataCopyParams params;
    params.blockLen = blockLen;
    params.blockCount = blockCount;
    params.srcStride = srcInnerLength;
    params.dstStride = dstInnerLength;
    DataCopy(dst, src, params);
}

} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif