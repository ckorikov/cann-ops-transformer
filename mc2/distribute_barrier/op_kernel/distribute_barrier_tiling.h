/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

/*!
 * \file distribute_barrier_tiling.h
 * \brief
 */
#ifndef DISTRIBUTE_BARRIER_TILING_H
#define DISTRIBUTE_BARRIER_TILING_H
#include "kernel_tiling/kernel_tiling.h"

struct DistributeBarrierInfo {
    uint32_t worldSize;
    uint32_t rankId;
    uint32_t aivNum;                     // aivNum
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct DistributeBarrierTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    DistributeBarrierInfo distributeBarrierInfo;
};

#endif // DISTRIBUTE_BARRIER_TILING_H