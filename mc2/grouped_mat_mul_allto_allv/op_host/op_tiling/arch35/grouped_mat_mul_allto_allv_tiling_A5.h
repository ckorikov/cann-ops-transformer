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
 * \file grouped_mat_mul_allto_allv_tiling_A5.h
 * \brief
 */

#ifndef __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_A5_H__
#define __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_A5_H__

#include "../grouped_mat_mul_allto_allv_tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/mc2_tiling_struct.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {

constexpr uint32_t MAX_EXPERT_SIZE = 512U;

BEGIN_TILING_DATA_DEF(GMMATAVACTiling)
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_EXPERT_SIZE, sendCnt);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_EXPERT_SIZE, recvCnt);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMATAVACTilingOp, GMMATAVACTiling)

BEGIN_TILING_DATA_DEF(GMMATAVTiling)
TILING_DATA_FIELD_DEF(uint64_t, A);
TILING_DATA_FIELD_DEF(uint64_t, H);
TILING_DATA_FIELD_DEF(uint64_t, sharedMatmulH);
TILING_DATA_FIELD_DEF(uint64_t, E_ep);
TILING_DATA_FIELD_DEF(uint64_t, N1);
TILING_DATA_FIELD_DEF(uint64_t, Bs);
TILING_DATA_FIELD_DEF(uint64_t, N2);
TILING_DATA_FIELD_DEF(uint64_t, BsK);
TILING_DATA_FIELD_DEF(uint64_t, epWorldSize);
TILING_DATA_FIELD_DEF(uint64_t, aivCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, aicCoreNum);
TILING_DATA_FIELD_DEF(bool, isGmmWeightTrans);
TILING_DATA_FIELD_DEF(bool, isMmWeightTrans);
TILING_DATA_FIELD_DEF(bool, isOptionalMatmul);
TILING_DATA_FIELD_DEF(bool, isOptionaSendRecvCountTensors);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMATAVTilingOp, GMMATAVTiling)

BEGIN_TILING_DATA_DEF(GroupedMatMulAlltoAllvTilingDataA5)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfgATA);
TILING_DATA_FIELD_DEF_STRUCT(GMMATAVTiling, commonTilingInfo);
TILING_DATA_FIELD_DEF_STRUCT(GMMATAVACTiling, aicpuTilingInfo);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, sharedExpMatmulTiling);
END_TILING_DATA_DEF;
// Register for all GroupedMatMulAlltoAllv but only used by A5.
REGISTER_TILING_DATA_CLASS(GroupedMatMulAlltoAllv, GroupedMatMulAlltoAllvTilingDataA5);

class GmmAlltoAllvTilingA5 : public GmmAlltoAllvTilingBase
{
public:
    explicit GmmAlltoAllvTilingA5(gert::TilingContext* context) : GmmAlltoAllvTilingBase(context){};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};

} // namespace optiling
#endif // __GROUPED_MAT_MUL_ALLTO_ALLV_TILING_A5_H__
