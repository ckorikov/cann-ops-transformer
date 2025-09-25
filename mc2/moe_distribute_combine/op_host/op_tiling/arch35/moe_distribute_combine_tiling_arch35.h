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
 * \file moe_distribute_combine_tiling_arch35.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_A5_H
#define MOE_DISTRIBUTE_COMBINE_TILING_A5_H

#include "tiling/moe_tiling_base.h"
#include "../moe_distribute_combine_tiling_helper.h"

namespace optiling {

ge::graphStatus MoeDistributeCombineTilingImpl(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(MoeDistributeCombineInfo)
    TILING_DATA_FIELD_DEF(uint32_t, epWorldSize);
    TILING_DATA_FIELD_DEF(uint32_t, tpWorldSize);
    TILING_DATA_FIELD_DEF(uint32_t, epRankId);
    TILING_DATA_FIELD_DEF(uint32_t, tpRankId);
    TILING_DATA_FIELD_DEF(uint32_t, expertShardType);
    TILING_DATA_FIELD_DEF(uint32_t, sharedExpertRankNum);
    TILING_DATA_FIELD_DEF(uint32_t, moeExpertNum);
    TILING_DATA_FIELD_DEF(uint32_t, moeExpertPerRankNum);
    TILING_DATA_FIELD_DEF(uint32_t, globalBs);
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint64_t, totalUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, totalWinSize);
    TILING_DATA_FIELD_DEF(uint32_t, hasSharedExpertX);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeDistributeCombineInfoOp, MoeDistributeCombineInfo);

BEGIN_TILING_DATA_DEF(MoeDistributeCombineTilingDataA5)
    TILING_DATA_FIELD_DEF(uint32_t, version);
    TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
    TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
    TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfgATA);
    TILING_DATA_FIELD_DEF_STRUCT(MoeDistributeCombineInfo, combineTilingInfo);
END_TILING_DATA_DEF;
// Register for all but only used by A5.
REGISTER_TILING_DATA_CLASS(MoeDistributeCombine, MoeDistributeCombineTilingDataA5);
REGISTER_TILING_DATA_CLASS(MoeDistributeCombineV2, MoeDistributeCombineTilingDataA5);

class MoeDistributeCombineTilingA5 : public MoeTilingBase {
public:
    explicit MoeDistributeCombineTilingA5(gert::TilingContext *context) : MoeTilingBase(context) {};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;

    MoeDistributeCombineTilingDataA5 tilingData_;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_COMBINE_TILING_A5_H