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
#ifndef ARCH35_CATLASS_KERNEL_DAVID_WQBMM_LOAD_IN_ADVANCE_H
#define ARCH35_CATLASS_KERNEL_DAVID_WQBMM_LOAD_IN_ADVANCE_H

#include "../utils/device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
template <typename ProblemShape, typename BlockMainloop, typename TileScheduler>
class wqbmmv2
{
public:
    struct Params {
        ProblemShape problemShape;
        typename BlockMainloop::Params mainloop;
        typename BlockMainloop::TileShapeL1 tileShape;
    };

public:
    DEVICE void operator()(Params const& params)
    {
        BlockMainloop blockMainloop;
        blockMainloop.Init(params.problemShape, params.mainloop);

        auto states = typename BlockMainloop::PipelineStateTuple{};
        auto pipelines = typename BlockMainloop::PipelineTuple{};

        TileScheduler tileScheduler(params.problemShape, params.mainloop);

        decltype(auto) mIterLoad = tileScheduler.MIter();
        decltype(auto) nIterLoad = tileScheduler.NIter();
        decltype(auto) mBiasIterLoad = tileScheduler.MIter();
        decltype(auto) nBiasIterLoad = tileScheduler.NIter();
        decltype(auto) itersLoad = AscendC::Std::tie(mIterLoad, nIterLoad, mBiasIterLoad, nBiasIterLoad);
        if (tileScheduler.IsValid()) {
            blockMainloop.LoadInAdvance(params.problemShape, pipelines, states, params.mainloop, itersLoad);
        }

        decltype(auto) mIterCompute = tileScheduler.MIterRef();
        decltype(auto) nIterCompute = tileScheduler.NIterRef();
        decltype(auto) mBiasIterCompute = tileScheduler.MIter();
        decltype(auto) itersCompute = AscendC::Std::tie(mIterCompute, nIterCompute, mBiasIterCompute, nIterCompute);

        // 尾核不执行Process
        while (tileScheduler.IsValid()) {
            blockMainloop.Process(params.problemShape, pipelines, states, params.mainloop, itersLoad, itersCompute);
            tileScheduler.FetchNextWork();
        }

        blockMainloop.ClearPipeline(pipelines, states);
    }
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif