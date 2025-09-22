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
#ifndef ARCH35_CATLASS_DISPATCH_POLICY_H
#define ARCH35_CATLASS_DISPATCH_POLICY_H

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
struct KernelWqbmm {
};

template <
    int Stages_, typename TileShapeUb_, typename TileShapeReg_, int32_t CoreType_, int32_t SubBlockNum_,
    uint8_t StageWeightIn_, uint8_t StageVfOut_, int32_t Kub_, typename KernelSchedule = KernelWqbmm>
struct MainloopDavidWqbmmUbAntiquantScmc {
    constexpr static int Stages = Stages_;
    constexpr static int32_t SubBlockNum = SubBlockNum_;
    using TileShapeUb = TileShapeUb_;
    using TileShapeReg = TileShapeReg_;
    uint8_t StageWeightIn = StageWeightIn_;
    uint8_t StageVfOut = StageVfOut_;
    int32_t Kub = Kub_;
    constexpr static int32_t CoreType = CoreType_;
    using Schedule = KernelSchedule;
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass

#endif