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
#ifndef ARCH35_CATLASS_PIPELINE_UTILS_H
#define ARCH35_CATLASS_PIPELINE_UTILS_H

#include "../utils/device_utils.h"
#include "kernel_event.h"

using AscendC::HardEvent;
using AscendC::Hardware;

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
namespace detail {
template <Hardware Src, Hardware Dst, bool FwdDirect>
DEVICE constexpr HardEvent GetQueEvt()
{
    if (Src == Hardware::GM) {
        if (Dst == Hardware::UB) {
            return FwdDirect ? HardEvent::MTE2_V : HardEvent::V_MTE2;
        } else if (Dst == Hardware::L1) {
            return FwdDirect ? HardEvent::MTE2_MTE1 : HardEvent::MTE1_MTE2;
        }
    } else if (Src == Hardware::UB) {
        if (Dst == Hardware::L1 || Dst == Hardware::GM) {
            return FwdDirect ? HardEvent::V_MTE3 : HardEvent::MTE3_V;
        }
    }
    return HardEvent::MAX;
}
} // namespace detail
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif