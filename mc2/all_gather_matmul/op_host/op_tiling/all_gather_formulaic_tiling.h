/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file all_gather_formulaic_tiling.h
 * \brief
 */
#ifndef __ALL_GATHER_FORMULAIC_TILING_H__
#define __ALL_GATHER_FORMULAIC_TILING_H__

#pragma once
#include "tiling/hccl_formulaic_tiling.h"
constexpr uint64_t LARGE_K_BOUNDARY = 8192;
constexpr uint64_t LARGE_N_BOUNDARY = 5120;
constexpr uint64_t SMALL_N_BOUNDARY = 2048;
constexpr uint64_t TINY_M = 512;
constexpr uint64_t MEDIAN_M = 4096;
constexpr double gatherLargerNKCommGrowRatio1 = 3;
constexpr double gatherLargerNKCommGrowRatio2 = 1.5;
constexpr uint64_t HUGE_K_BOUNDARY = 32768;

class AllGatherPlusMM : public OneCalcOneCommBase
{
public:
    double frontMMTime_ = 0;
    bool strongTpBound_ = false;
    bool hasLocalAtFront_ = true;  // local提前计算

    // Constructor
    explicit AllGatherPlusMM(const mc2tiling::TilingArgs& args, uint32_t inputRankDim, KernelType inputKernelType,
                             SocVersion inputSocVersion = SocVersion::SOC910_B)
        : OneCalcOneCommBase(args, inputRankDim, inputKernelType, inputSocVersion)
    {
        commPerf_.SetCommShapeLen(clusterInfo_.kValue);
        commPerf_.SetCommDTypeSize(clusterInfo_.inMatrixADtypeSize);
        rankTileNum_ = commPerf_.GetRankTileNum();
        tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());

        if (clusterInfo_.socType == SocVersion::SOC910_B) {
            tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankDim_));
        } else {
            tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
        }
    }
    void EstimateKernelTime() override;
    void SelectTilingMethod() override;

private:
    void SetCommTimeFactorForA5();
    void SetCommTimeFactorForOther();
    void SetCommTimeFactor();
    void PrintEstimateKernelTimeResult(double totalMatmulTime, double totalTpTime);
};

#endif //__ALL_GATHER_FORMULAIC_TILING_H__
