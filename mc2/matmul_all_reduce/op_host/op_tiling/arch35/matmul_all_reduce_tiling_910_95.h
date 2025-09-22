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

/*!
 * \file matmul_all_reduce_tiling_910_95.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_910_95_H
#define MATMUL_ALL_REDUCE_TILING_910_95_H

#include "../matmul_all_reduce_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAllReduce910TilingDataA5)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfg);
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000000001, MatmulAllReduce910TilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000001100, MatmulAllReduce910TilingDataA5);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_11000000000000000009, MatmulAllReduce910TilingDataA5);

class MatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase
{
    friend class TilingTransferHelperA5;

public:
    explicit MatmulAllReduceTilingA5(gert::TilingContext* context);
    MatmulAllReduceTilingA5(gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, MatmulAllReduce910TilingDataA5* out);
    ~MatmulAllReduceTilingA5() override = default;

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

    void PrintExtendMatmulTiling(bool isTail) override;
    void DoEmptyTensorTiling() override;
    void SetMc2Hcomm();
    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus CheckAxisSize();
    ge::graphStatus CheckX1X2();
    MatmulAllReduce910TilingDataA5 matmulAllReduce910TilingDataSelf_;
    MatmulAllReduce910TilingDataA5& matmulAllReduce910TilingData_;
    uint64_t myWorkSpaceSize_{0U};
};

class TilingTransferHelperA5 : public matmul_v3::MatmulV3BaseTiling
{
public:
    TilingTransferHelperA5(MatmulAllReduceTilingA5& matmulAllReduceTiling910, MatmulTilingData& data)
        : MatmulV3BaseTiling(matmulAllReduceTiling910.context_, &data), tilingProcesser_(matmulAllReduceTiling910)
    {}

    ge::graphStatus GetShapeAttrsInfo() override
    {
        auto&& tilingArgs = tilingProcesser_.args_;
        args_.opName = tilingProcesser_.opName_;
        args_.isATrans = tilingArgs.isATrans;
        args_.isBTrans = tilingArgs.isBTrans;
        args_.hasBias = tilingArgs.isBias;
        args_.aType = tilingArgs.geAType;
        args_.bType = tilingArgs.geBType;
        args_.cType = tilingArgs.geCType;
        args_.biasType = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
        args_.aFormat = ge::FORMAT_ND;
        args_.outFormat = ge::FORMAT_ND;
        args_.mValue = tilingArgs.mValue;
        args_.kValue = tilingArgs.kValue;
        args_.nValue = tilingArgs.nValue;
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus PostTiling() override
    {
        tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
        OP_LOGI(tilingProcesser_.opName_, "Set mm workspace size=%lu to mc2", tilingProcesser_.myWorkSpaceSize_);
        return ge::GRAPH_SUCCESS;
    }

private:
    MatmulAllReduceTilingA5& tilingProcesser_;
};
} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_910_95_H