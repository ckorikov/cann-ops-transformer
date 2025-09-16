/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
 * \file moe_distribute_dispatch_tiling_arch35.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_ARCH35_H
#define MOE_DISTRIBUTE_DISPATCH_TILING_ARCH35_H

#include "tiling/moe_tiling_base.h"
#include "../moe_distribute_dispatch_tiling_helper.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MoeDistributeDispatchInfo)
    TILING_DATA_FIELD_DEF(uint32_t, epWorldSize);
    TILING_DATA_FIELD_DEF(uint32_t, tpWorldSize);
    TILING_DATA_FIELD_DEF(uint32_t, epRankId);
    TILING_DATA_FIELD_DEF(uint32_t, tpRankId);
    TILING_DATA_FIELD_DEF(uint32_t, expertShardType);
    TILING_DATA_FIELD_DEF(uint32_t, sharedExpertRankNum);               
    TILING_DATA_FIELD_DEF(uint32_t, moeExpertNum);
    TILING_DATA_FIELD_DEF(uint32_t, quantMode);
    TILING_DATA_FIELD_DEF(uint32_t, globalBs);
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(bool, isQuant);
    TILING_DATA_FIELD_DEF(bool, reserved1);
    TILING_DATA_FIELD_DEF(bool, reserved2);
    TILING_DATA_FIELD_DEF(bool, reserved3);
    TILING_DATA_FIELD_DEF(uint64_t, totalUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, totalWinSize);
    TILING_DATA_FIELD_DEF(uint32_t, expertTokenNumsType);
    TILING_DATA_FIELD_DEF(uint64_t, scalesRow);
    TILING_DATA_FIELD_DEF(uint64_t, scalesCol);
    TILING_DATA_FIELD_DEF(uint32_t, scalesTypeSize);
    TILING_DATA_FIELD_DEF(uint64_t, scalesCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeDistributeDispatchInfoOp, MoeDistributeDispatchInfo);    

BEGIN_TILING_DATA_DEF(MoeDistributeDispatchTilingDataA5)
    TILING_DATA_FIELD_DEF(uint32_t, version);
    TILING_DATA_FIELD_DEF(uint32_t, hcommCnt);
    TILING_DATA_FIELD_DEF_STRUCT(MC2ServerCfg, serverCfg);
    TILING_DATA_FIELD_DEF_STRUCT(MC2HcommCfg, hcommCfgATA);
    TILING_DATA_FIELD_DEF_STRUCT(MoeDistributeDispatchInfo, dispatchTilingInfo);
END_TILING_DATA_DEF;
// Register for all but only used by A5.
REGISTER_TILING_DATA_CLASS(MoeDistributeDispatch, MoeDistributeDispatchTilingDataA5);

class MoeDistributeDispatchTilingA5 : public MoeTilingBase {
public:
    explicit MoeDistributeDispatchTilingA5(gert::TilingContext *context) : MoeTilingBase (context) {};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;

    MoeDistributeDispatchTilingDataA5 tilingData_;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_DISPATCH_TILING_ARCH35_H