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
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief
 */

#ifndef __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__
#define __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t MAX_EXPERT_SIZE = 512U; // 最大通信域专家的数量

struct GmmAlltoAllvAicpuTiling {
    uint16_t sendCnt[MAX_EXPERT_SIZE];
    uint16_t recvCnt[MAX_EXPERT_SIZE];
};

struct GmmAlltoAllvCommonTilingInfo {
    uint64_t A;
    uint64_t H;
    uint64_t sharedMatmulH;
    uint64_t E_ep;
    uint64_t N1;
    uint64_t Bs;
    uint64_t N2;
    uint64_t BsK;
    uint64_t epWorldSize;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    bool isGmmWeightTrans;
    bool isMmWeightTrans;
    bool isOptionalMatmul;
    bool isOptionalSendRecvCountTensors;
};

class GroupedMatMulAlltoAllvTilingData
{
public:
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling alltoAllvCcTiling;
    GmmAlltoAllvCommonTilingInfo commonTilingInfo;
    TCubeTiling matmulTiling;
    TCubeTiling sharedExpMatmulTiling;
    GmmAlltoAllvAicpuTiling aicpuTilingInfo;
};

#endif // __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__