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
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_BASE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_BASE_H

#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "kernel_event.h"
#include "utils.h"

using AscendC::HardEvent;
using AscendC::Hardware;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
template <Hardware Src, Hardware Dst, uint8_t Stages_, typename Derived>
struct PipelineStageSingleCoreBase {
    static constexpr uint8_t Stages = Stages_;
    static constexpr HardEvent ForwardHardEvent = detail::GetQueEvt<Src, Dst, true>();
    static constexpr HardEvent BackwardHardEvent = detail::GetQueEvt<Src, Dst, false>();
    TEventID backwardEventId[Stages];

    DEVICE
    PipelineStageSingleCoreBase()
    {}

    DEVICE
    void ProducerWait(PipelineState<Stages> const& state) const
    {
        ProducerWait(state.index());
    }

    DEVICE
    void ProducerRelease(PipelineState<Stages> const& state) const
    {
        static_cast<const Derived*>(this)->ProducerRelease(state.index());
    }

    DEVICE
    void ConsumerWait(PipelineState<Stages> const& state) const
    {
        static_cast<const Derived*>(this)->ConsumerWait(state.index());
    }

    DEVICE
    void ConsumerRelease(PipelineState<Stages> const& state) const
    {
        ConsumerRelease(state.index());
    }

    DEVICE
    void ProducerWait(uint64_t index) const
    {
        WaitFlag<BackwardHardEvent>(backwardEventId[index]);
    }

    DEVICE
    void ConsumerRelease(uint64_t index) const
    {
        SetFlag<BackwardHardEvent>(backwardEventId[index]);
    }

    DEVICE void Clear() const
    {
        // 多set需要wait掉
        #pragma unroll
        for (uint8_t i = 0; i < Stages; ++i) {
            ProducerWait(backwardEventId[i]);
        }
    }
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif