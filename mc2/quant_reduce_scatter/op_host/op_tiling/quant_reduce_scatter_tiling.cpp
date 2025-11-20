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
 * \file quant_reduce_scatter_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "quant_reduce_scatter_check_tiling.h"
#include "../../op_kernel/quant_reduce_scatter_tiling_key.h"
#include "../../op_kernel/quant_reduce_scatter_tiling_data.h"

using namespace AscendC;
using namespace ge;

namespace MC2Tiling {
constexpr size_t X_INDEX = 0;
constexpr size_t SCALE_INDEX = 1;
constexpr size_t GROUP_INDEX = 0;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t RANK_SIZE_NUMBER_TWO = 2;
constexpr uint32_t RANK_SIZE_NUMBER_FOUR = 4;
constexpr uint32_t RANK_SIZE_NUMBER_EIGHT = 8;
constexpr uint32_t DEFAULT_BLOCK_DIM = 1U;
const std::string OP_NAME = "QuantReduceScatter";
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t AIV_TYPE = 3;

static ge::graphStatus CheckSocVersion(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 校验socVersion
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE(nodeName, "platformInfoPtr is null."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND910_95,
        OP_LOGE(nodeName, "SocVersion needed to be 910_95."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetRankSize(const std::string &group, QuantReduceScatterTilingParams &params,
                                           const char *nodeName)
{
    uint32_t rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group.c_str());
    params.rankSize = rankSize;
    OP_TILING_CHECK(rankSize != RANK_SIZE_NUMBER_TWO &&
                    rankSize != RANK_SIZE_NUMBER_FOUR &&
                    rankSize != RANK_SIZE_NUMBER_EIGHT,
        OP_LOGE(nodeName, "The rankSize should be 2,4,8, actual rankSize is %u.", rankSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext *context, QuantReduceScatterTilingData *tiling,
                        const std::string &group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "QuantReduceScatter group = %s", group.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);  // MTE方式必要适配
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
}

static void SetTilingDataAndBlockDim(gert::TilingContext *context, QuantReduceScatterTilingData &tilingData)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    tilingData.quantReduceScatterTilingInfo.aivNum = aivNum;

    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scaleValueTwo = context->GetInputShape(SCALE_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    tilingData.quantReduceScatterTilingInfo.bs = xValueOne;
    tilingData.quantReduceScatterTilingInfo.hiddenSize = xValueTwo;
    tilingData.quantReduceScatterTilingInfo.scaleHiddenSize = scaleValueTwo;
}

static void SetTilingKey(gert::TilingContext *context)
{
    uint32_t quantReduceScatterTemplateId = MTE_COMM; 
    const char *nodeName = context->GetNodeName();
    // 设置tilingKey模板参数方式
    const uint64_t tilingKey = GET_TPL_TILING_KEY(quantReduceScatterTemplateId);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu]", tilingKey);
}

static ge::graphStatus QuantReduceScatterTilingFunc(gert::TilingContext* context)
{
    // 1.tiling校验
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(OP_NAME, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    QuantReduceScatterTilingData* tilingData = context->GetTilingData<QuantReduceScatterTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tlingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter QuantReduceScatter tiling check func.");
    // 校验socVersion与Attr属性
    OP_TILING_CHECK(CheckSocVersion(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "socVersion is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(QuantReduceScatterCheckTiling::CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Attrs are invalied."), return ge::GRAPH_FAILED);
    // 获取group以便ranksize设置与通信设置
    std::string group = "";
    const char *groupPtr = context->GetAttrs()->GetAttrPointer<char>(GROUP_INDEX);
    group = std::string(groupPtr);
    // 校验ranksize
    QuantReduceScatterTilingParams params;
    OP_TILING_CHECK(CheckAndSetRankSize(group, params, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "RankSize is invalied."), return ge::GRAPH_FAILED);
    // 校验输入输出tensor的dim/dtype/format
    OP_TILING_CHECK(QuantReduceScatterCheckTiling::TilingCheckQuantReduceScatter(context, params) !=
        ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);
    // 校验WinSize
    OP_TILING_CHECK(!QuantReduceScatterCheckTiling::CheckWindowSize(context, params),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL."), return ge::GRAPH_FAILED);

    // 2.tiling设置
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, group);
    SetTilingDataAndBlockDim(context, *tilingData);
    SetTilingKey(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantReduceScatter)
    .Tiling(QuantReduceScatterTilingFunc);
} // namespace MC2Tiling
