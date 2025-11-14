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
 * \file all_gather_add_tiling.cc
 * \brief
 */
#include "vector"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "ops_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/all_gather_add_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
    constexpr uint32_t TILE_NUM = 1;
    constexpr uint32_t COMM_TURN = 1;
}
namespace optiling {

static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context)
{
    OP_TILING_CHECK(mc2tiling::Mc2TilingUtils::CommonParamCheck(context) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "common check failed"), return ge::GRAPH_FAILED);

    const gert::StorageShape* aShape = context->GetInputShape(0);
    uint64_t valueOne = aShape->GetStorageShape().GetDim(0);
    uint64_t valueTwo = aShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(valueOne == 0 || valueTwo == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "the value is invalid"), return ge::GRAPH_FAILED);
    
    if (context->GetAttrs() == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get attrs failed");
    } 
    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OP_TILING_CHECK(group == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group is nullptr. "),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 设置hccl高阶API Tiling结构体
static void InitHcclParam(AllGatherAddTilingData* tilingData, const char* group)
{
    std::string algConfig = "AllGather=level0:fullmesh";
    Mc2CcTilingConfig mc2CcTilingConfig(group, HCCL_CMD_ALLGATHER, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
}

static ge::graphStatus AllGatherAddTilingFunc(gert::TilingContext *context) {
    // 对参数进行校验
    OP_TILING_CHECK(AllGatherParamsCheck(context) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "param is invalid"), return ge::GRAPH_FAILED);
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAiv());

    // 设置TilingData
    AllGatherAddTilingData* tilingData = context->GetTilingData<AllGatherAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(
        memset_s(tilingData, sizeof(AllGatherAddTilingData), 0, sizeof(AllGatherAddTilingData)) != EOK,
        OP_LOGE(context, "set AllGatherAdd tiling data error"), return ge::GRAPH_FAILED);
    auto dataType = context->GetInputTensor(1)->GetDataType();
    tilingData->commTurn = COMM_TURN;
    tilingData->tileNum = TILE_NUM;
    tilingData->totalLength = context->GetInputTensor(1)->GetShapeSize(); // 总长度是参与Add操作的数据个数
    tilingData->blockLength = tilingData->totalLength / context->GetBlockDim(); // 每个核需要计算的数据个数
    tilingData->tileLength = tilingData->blockLength / tilingData->tileNum; // 每个核内每个数据块的数据个数
    tilingData->gatherTileLength = tilingData->totalLength / 2; // 待gather的数据个数
    
    // 设置workspaceSize gather out需要额外的临时内存，大小=input b
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context,currentWorkspace);
    // 如需使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace大小
    uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 预留18M + gather_out, gather_out 大小跟x1输入一样
    currentWorkspace[0] = sysWorkSpaceSize + tilingData->totalLength;

    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    InitHcclParam(tilingData, group);
    return ge::GRAPH_SUCCESS;
}

struct AllGatherAddCompileInfo {};

static ge::graphStatus TilingParseForAllGatherAdd([[maybe_unused]] gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherAdd)
    .Tiling(AllGatherAddTilingFunc)
    .TilingParse<AllGatherAddCompileInfo>(TilingParseForAllGatherAdd);
}  // namespace optiling