/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_add.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "all_gather_add.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void all_gather_add_custom(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
    GM_ADDR gatherGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 注册算子Tiling结构体
    REGISTER_TILING_DEFAULT(AllGatherAddTilingData);
    auto tiling = (__gm__ AllGatherAddTilingData*)tilingGM;
    __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
    __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
    GET_TILING_DATA(tilingData, tilingGM);
    Tpipe pipe;

    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

    // 初始化Add对象并对本卡数据进行Add计算，固定shape
    AllGatherAdd allGatherAdd;
    allGatherAdd.Init(aGM, bGM, cGM, workspaceGM, contextGM, &tilingData, &pipe);
    allGatherAdd.Process();
}