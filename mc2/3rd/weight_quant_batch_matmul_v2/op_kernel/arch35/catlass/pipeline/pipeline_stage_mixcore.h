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
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_MIXCORE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_MIXCORE_H

#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "kernel_event.h"
#include "pipeline_state.h"
#include "utils.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {

#if defined(__CCE_KT_TEST__)
template <uint8_t modeId, pipe_t pipe>
DEVICE void CrossCoreWaitFlag(uint16_t flagId)
{
    auto strPipe = detail::PipelineToStr(pipe);
    X_LOG("CrossCoreWaitFlag<%d,%s>(%d)", modeId, strPipe.c_str(), flagId);
}

template <uint8_t modeId, pipe_t pipe>
DEVICE void CrossCoreSetFlag(uint16_t flagId)
{
    auto strPipe = detail::PipelineToStr(pipe);
    X_LOG("CrossCoreSetFlag<%d,%s>(%d)", modeId, strPipe.c_str(), flagId);
}
#endif

template <
    pipe_t ProducerPipeline, pipe_t ConsumerPipeline, int32_t SubBlockDim, uint8_t SyncMode = 4, uint16_t FlagId = 1>
struct PipelineStageMixCore {
    static constexpr uint64_t FLAG_ID_MAX = 16;

    template <uint8_t Stages>
    DEVICE static void ProducerWaitAfterStages(PipelineState<Stages> const& state)
    {
        if (state.count() >= Stages) {
            CrossCoreWaitFlag<SyncMode, ProducerPipeline>(FlagId);
        }
    }

    DEVICE
    static void ProducerRelease()
    {
        CrossCoreSetFlag<SyncMode, ProducerPipeline>(FlagId);
    }

    DEVICE
    static void ConsumerWait()
    {
        if constexpr (SubBlockDim == 2) {
            CrossCoreWaitFlag<SyncMode, ConsumerPipeline>(FlagId + FLAG_ID_MAX);
        }
        CrossCoreWaitFlag<SyncMode, ConsumerPipeline>(FlagId);
    }

    DEVICE
    static void ConsumerRelease()
    {
        if constexpr (SubBlockDim == 2) {
            CrossCoreSetFlag<SyncMode, ConsumerPipeline>(FlagId + FLAG_ID_MAX);
        }
        CrossCoreSetFlag<SyncMode, ConsumerPipeline>(FlagId);
    }

    template <uint8_t Stages>
    DEVICE static void Clear(PipelineState<Stages> const& state)
    {
        if ASCEND_IS_AIV {
            for (uint8_t i = 0; i < Min(static_cast<uint64_t>(Stages), state.count()); ++i) {
                ProducerWait();
            }
        }
    }

private:
    DEVICE
    static void ProducerWait()
    {
        CrossCoreWaitFlag<SyncMode, ProducerPipeline>(FlagId);
    }
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif