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
 * \file moe_distribute_dispatch_v2_full_mesh.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch/moe_distribute_base.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_dispatch/check_winsize.h"

namespace MoeDistributeDispatchV2FullMeshImpl {

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateMC2TypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeDispatchV2FullMesh {
public:
    __aicore__ inline MoeDistributeDispatchV2FullMesh() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, 
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, 
                                GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
};


template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR expandXOut, 
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::Process()
{
}

} // MoeDistributeDispatchV2FullMeshImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H