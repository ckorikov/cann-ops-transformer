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
 * \file grouped_matmul_finalize_routing_tiling.cc
 * \brief
 */
#include "lower_triangular_inverse_tiling.h"
#include "lower_triangular_inverse_base_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"

using namespace optiling::LowerTriangularInverse;

namespace optiling {

using namespace Ops::Transformer::OpTiling;

REGISTER_TILING_TEMPLATE("LowerTriangularInverse", LowerTriangularInverseBaseTiling, 0);

static ge::graphStatus LowerTriangularInverseTilingFunc(gert::TilingContext* context)
{
     OP_CHECK_IF(context == nullptr,
        OPS_REPORT_CUBE_INNER_ERR("[LowerTriangularInverseTilingFunc]", " context is null"),
        return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForLowerTriangularInverse(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("[TilingPrepareForLowerTriangularInverse]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<LowerTriangularInverseCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL; // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_LOGI(context->GetNodeName(),
            "parse compile info success soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
            static_cast<int>(compileInfoPtr->socVersion),
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicNum,
            compileInfoPtr->supportL0c2out,
            compileInfoPtr->supportL12BtBf16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LowerTriangularInverse)
    .Tiling(LowerTriangularInverseTilingFunc)
    .TilingParse<LowerTriangularInverseCompileInfo>(TilingPrepareForLowerTriangularInverse);
}
