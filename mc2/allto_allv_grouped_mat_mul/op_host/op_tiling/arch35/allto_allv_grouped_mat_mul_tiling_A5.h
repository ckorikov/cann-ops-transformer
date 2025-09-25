/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_tiling_A5.h
 * \brief
 */

#ifndef __ALLTO_ALLV_GROUPED_MAT_MUL_TILING_A5_H__
#define __ALLTO_ALLV_GROUPED_MAT_MUL_TILING_A5_H__

#include <cstdint>
#include "tiling/matmul_formulaic_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/mc2_tiling_struct.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "../allto_allv_grouped_mat_mul_tiling_base.h"

namespace optiling {

constexpr uint32_t MAX_EXPERT_SIZE = 256U;
constexpr uint32_t MAX_EP_RANK_SIZE = 64U;

BEGIN_TILING_DATA_DEF(ATAVGMMACTiling)
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_EXPERT_SIZE, sendCnt);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_EXPERT_SIZE, recvCnt);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ATAVGMMACTilingOp, ATAVGMMACTiling);

BEGIN_TILING_DATA_DEF(AlltoAllvGmmCommonTilingInfo)
TILING_DATA_FIELD_DEF(uint64_t, BSK);
TILING_DATA_FIELD_DEF(uint64_t, BS);
TILING_DATA_FIELD_DEF(uint64_t, K);
TILING_DATA_FIELD_DEF(uint64_t, H1);
TILING_DATA_FIELD_DEF(uint64_t, H2);
TILING_DATA_FIELD_DEF(uint64_t, A);
TILING_DATA_FIELD_DEF(uint64_t, N1);
TILING_DATA_FIELD_DEF(uint64_t, N2);
TILING_DATA_FIELD_DEF(uint64_t, epWorldSize);
TILING_DATA_FIELD_DEF(uint64_t, stepSize);
TILING_DATA_FIELD_DEF(uint64_t, E_ep);
TILING_DATA_FIELD_DEF(uint64_t, commOut);
TILING_DATA_FIELD_DEF(uint64_t, aivCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, aicCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalUbSize);
TILING_DATA_FIELD_DEF(bool, isGmmWeightTrans);
TILING_DATA_FIELD_DEF(bool, isMmWeightTrans);
TILING_DATA_FIELD_DEF(bool, isSendCntsTensor);
TILING_DATA_FIELD_DEF(bool, isRecvCntsTensor);
TILING_DATA_FIELD_DEF(bool, isPermuteOut);
TILING_DATA_FIELD_DEF(bool, isNeedMM);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AlltoAllvGmmCommonTilingInfoOp, AlltoAllvGmmCommonTilingInfo);

BEGIN_TILING_DATA_DEF(AlltoAllvGroupedMatMulTilingDataA5)
TILING_DATA_FIELD_DEF(uint32_t, version);
TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfgATA);
TILING_DATA_FIELD_DEF_STRUCT(AlltoAllvGmmCommonTilingInfo, commonTilingInfo);
TILING_DATA_FIELD_DEF_STRUCT(ATAVGMMACTiling, aicpuTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gmmTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;
// Register for all but only used by A5.
REGISTER_TILING_DATA_CLASS(AlltoAllvGroupedMatMul, AlltoAllvGroupedMatMulTilingDataA5);

class AlltoAllvGmmTilingA5 : public AlltoAllvGmmTilingBase
{
public:
    explicit AlltoAllvGmmTilingA5(gert::TilingContext* context) : AlltoAllvGmmTilingBase(context){};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;

    AlltoAllvGroupedMatMulTilingDataA5 tilingData_;
};

} // namespace optiling
#endif // __ALLTO_ALLV_GROUPED_MAT_MUL_TILING_A5_H__
