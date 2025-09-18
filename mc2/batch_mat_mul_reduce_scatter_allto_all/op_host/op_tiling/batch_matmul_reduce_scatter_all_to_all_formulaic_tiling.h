/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file batch_matmul_reduce_scatter_all_to_all_formulaic_tiling.h
 * \brief
 */

#ifndef __BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_FORMULAIC_TILING_H__
#define __BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_FORMULAIC_TILING_H__

#pragma once
#include "inc/tiling/one_calc_two_comm_tiling.h"

class ReduceScatterAll2AllBMM : public OneCalcTwoCommBase {
public:
    explicit ReduceScatterAll2AllBMM(const mc2tiling::TilingArgs& args, uint64_t inputEpDim, uint64_t inputTpDim,
        uint64_t batchSize, SocVersion inputSocVersion = SocVersion::SOC910_93)
        : OneCalcTwoCommBase (args, inputEpDim, inputTpDim, batchSize, inputSocVersion)
    {
        epCommPerf.SetCommShapeLen(clusterInfo.nValue / tpDim);
        epCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
        tpCommPerf.SetCommShapeLen(clusterInfo.nValue);
        tpCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
    }
};

class ReduceScatterAll2AllBMMShardH : public OneCalcTwoCommShardHBase {
public:
    explicit ReduceScatterAll2AllBMMShardH(const mc2tiling::TilingArgs& args, uint64_t inputEpDim, uint64_t inputTpDim,
        uint64_t batchSize, SocVersion inputSocVersion = SocVersion::SOC910_93)
        : OneCalcTwoCommShardHBase (args, inputEpDim, inputTpDim, batchSize, inputSocVersion)
    {
        epCommPerf.SetCommShapeLen(clusterInfo.nValue);
        epCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
        tpCommPerf.SetCommShapeLen(clusterInfo.nValue);
        tpCommPerf.SetCommDTypeSize(clusterInfo.outMatrixCDtypeSize);
    }
    bool SetShortTilePositionFlag(double totalBmmTime, double totalCommTime) override {
        // short tile at front when totalBmmTime < totalCommTime
        if (totalBmmTime < totalCommTime) {
            return true;
        } else {
            return false;
        }
    };
};
#endif //__BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_FORMULAIC_TILING_H__