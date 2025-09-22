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
#ifndef ARCH35_CATLASS_TILE_COPY_GM_TO_L1_H
#define ARCH35_CATLASS_TILE_COPY_GM_TO_L1_H

#include "../utils/device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;

template <typename T>
DEVICE void CopyGmToL1IntervalDataCopy(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t nValue, uint32_t dValue, uint32_t dstInnerLength,
    uint32_t srcInnerLength)
{
    // ND2NZ
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = nValue;
    nd2nzParams.dValue = dValue;
    nd2nzParams.srcDValue = srcInnerLength;
    // PS tiling保证是16对齐的
    nd2nzParams.dstNzC0Stride = dstInnerLength;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(dst, src, nd2nzParams);
}

template <typename T>
DEVICE void CopyGmToL1(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t size)
{
    DataCopy(dst, src, size);
}

} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif