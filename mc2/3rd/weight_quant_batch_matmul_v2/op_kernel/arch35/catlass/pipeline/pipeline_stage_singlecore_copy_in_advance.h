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
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_COPY_IN_ADVANCE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_COPY_IN_ADVANCE_H

#include "../utils/math_utils.h"
#include "pipeline_stage_singlecore_base.h"
#include "pipeline_state.h"
#include "utils.h"

using AscendC::HardEvent;
using AscendC::Hardware;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

#if defined(__CCE_KT_TEST__)
#include <set>
std::set<std::tuple<HardEvent, int>> setCopyInAdvance;
#endif

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {

template <Hardware Src, Hardware Dst, uint8_t Stages_>
struct PipelineStageSingleCoreCopyInAdvance
    : public PipelineStageSingleCoreBase<Src, Dst, Stages_, PipelineStageSingleCoreCopyInAdvance<Src, Dst, Stages_>> {
    static constexpr uint8_t Stages = Stages_;
    using Base =
        PipelineStageSingleCoreBase<Src, Dst, Stages_, PipelineStageSingleCoreCopyInAdvance<Src, Dst, Stages_>>;
    TEventID forwardEventId[Stages];

    using Base::ConsumerWait;
    using Base::ProducerRelease;

    DEVICE
    PipelineStageSingleCoreCopyInAdvance()
    {
        #pragma unroll
        for (uint8_t i = 0; i < Stages; ++i) {
            forwardEventId[i] = GetTPipePtr()->AllocEventID<Base::ForwardHardEvent>();
        }
        #pragma unroll
        for (uint8_t i = 0; i < Stages; ++i) {
            Base::backwardEventId[i] = GetTPipePtr()->AllocEventID<Base::BackwardHardEvent>();
            Base::ConsumerRelease(i);
        }
    }

    DEVICE
    void ProducerRelease(uint64_t index) const
    {
        SetFlag<Base::ForwardHardEvent>(forwardEventId[index]);
    }

    DEVICE
    void ConsumerWait(uint64_t index) const
    {
        WaitFlag<Base::ForwardHardEvent>(forwardEventId[index]);
    }

    DEVICE
    void Clear(PipelineState<Stages> &state) const
    {
        #pragma unroll
        for (uint8_t i = 0; i < Stages - 1; ++i) {
            Base::ProducerWait(state.index());
            ++state;
        }
        ConsumerWait(state.index());
    }
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif