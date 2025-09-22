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
 * \file matmul_all_reduce_tiling_910.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_910_H
#define MATMUL_ALL_REDUCE_TILING_910_H

#include "matmul_all_reduce_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAllReduce910TilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000001100, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000009, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000000, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000000001, MatmulAllReduce910TilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce910TilingDataOp, MatmulAllReduce910TilingData);

class MatmulAllReduceTiling910 : public MatmulAllReduceTilingBase
{
    friend class MMNTilingTransferHelper;
    friend class TilingTransferHelper;
    friend class MatmulAllReduceAddRmsNormTiling;

public:
    explicit MatmulAllReduceTiling910(gert::TilingContext* context);
    MatmulAllReduceTiling910(gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, MatmulAllReduce910TilingData* out);
    ~MatmulAllReduceTiling910() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus Do910Tiling();

    Mc2Msg& MutableMc2MsgData() override;

    RCSTiling& MutableRCSTilingData() override;

    TCubeTiling& MutableTCubeTileTilingData() override;

    TCubeTiling& MutableTCubeTailTilingData() override;

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus CheckAxisSize();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputFormat();
    ge::graphStatus CheckInputShape();
    MatmulAllReduce910TilingData matmulAllReduce910TilingDataSelf_;
    MatmulAllReduce910TilingData& matmulAllReduce910TilingData_;
    uint64_t myWorkSpaceSize_{0U};
};

class TilingTransferHelper : public matmul_v3::MatmulV3BaseTiling
{
public:
    TilingTransferHelper(MatmulAllReduceTiling910& matmulAllReduceTiling910, MatmulTilingData& data);

    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;

private:
    MatmulAllReduceTiling910& tilingProcesser_;
};
} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_910_H