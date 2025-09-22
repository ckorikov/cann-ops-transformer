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
 * \file weight_quant_matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#include "matmul_all_reduce_tiling.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantBatchMatmulV2TilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantBatchMatmulV2TilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310100, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311100, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310110, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311110, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310300, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_310310, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311300, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_311310, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_810200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_811200, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_810210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_811210, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000008, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(WeightQuantMatmulAllReduceTilingDataOp, WeightQuantMatmulAllReduceTilingData);

constexpr int64_t ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;

class WeightQuantMatmulAllReduceTiling : public MatmulAllReduceTilingBase
{
    friend class WeightQuantTilingTransferHelper;
    friend class WeightQuantMatmulAllReduceAddRmsNormTiling;

public:
    explicit WeightQuantMatmulAllReduceTiling(gert::TilingContext* context);
    WeightQuantMatmulAllReduceTiling(
        gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, WeightQuantMatmulAllReduceTilingData* out);
    ~WeightQuantMatmulAllReduceTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg& MutableMc2MsgData() override
    {
        return weightQuantMatmulAllReduceTilingData_.msg;
    }

    RCSTiling& MutableRCSTilingData() override
    {
        return weightQuantMatmulAllReduceTilingData_.param;
    }

    TCubeTiling& MutableTCubeTileTilingData() override
    {
        return weightQuantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
    }

    TCubeTiling& MutableTCubeTailTilingData() override
    {
        return weightQuantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
    }

    ge::graphStatus DoWeightQuantTiling();

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus CheckAxisSize();
    WeightQuantMatmulAllReduceTilingData weightQuantMatmulAllReduceTilingDataSelf_;
    WeightQuantMatmulAllReduceTilingData& weightQuantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
};

class WeightQuantTilingTransferHelper : public WeightQuantBatchMatmulV2TilingCustom
{
public:
    WeightQuantTilingTransferHelper(
        WeightQuantMatmulAllReduceTiling& weightQuantMatmulAllReduceTiling, WeightQuantBatchMatmulV2TilingData& data)
        : WeightQuantBatchMatmulV2TilingCustom(weightQuantMatmulAllReduceTiling.context_, &data),
          tilingProcesser_(weightQuantMatmulAllReduceTiling)
    {}
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(WeightQuantBatchMatmulInfo& weightQuantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    WeightQuantMatmulAllReduceTiling& tilingProcesser_;
};
} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
