/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#include "../matmul_all_reduce_tiling_base.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(Mc2WeightQuantBatchMatmulV2TilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(Mc2WeightQuantBatchMatmulV2TilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365056114230017, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365330992136961, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365056651100929, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365331529007873, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365057187971841, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365332065878785, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365057724842753, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365332602749697, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365058261713665, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365058798584577, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365333139620609, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365333676491521, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365057188299521, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365332066206465, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365057725170433, WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_365332603077377, WeightQuantMatmulAllReduceTilingData);
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

class WeightQuantTilingTransferHelper : public Mc2WeightQuantBatchMatmulV2TilingCustom
{
public:
    WeightQuantTilingTransferHelper(
        WeightQuantMatmulAllReduceTiling& weightQuantMatmulAllReduceTiling, Mc2WeightQuantBatchMatmulV2TilingData& data)
        : Mc2WeightQuantBatchMatmulV2TilingCustom(weightQuantMatmulAllReduceTiling.context_, &data),
          tilingProcesser_(weightQuantMatmulAllReduceTiling)
    {}
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(Mc2WeightQuantBatchMatmulInfo& weightQuantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    WeightQuantMatmulAllReduceTiling& tilingProcesser_;
};
} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
