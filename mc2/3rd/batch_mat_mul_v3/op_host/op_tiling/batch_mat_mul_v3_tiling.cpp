/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file batch_mat_mul_v3_tiling.cc
 * \brief
 */
#include "batch_mat_mul_v3_tiling.h"

#include <type_traits>

#include "cube_tiling_runtime.h"
#include "matmul_v3_simplifiedkey.h"
#include "matmul_v3_platform_common.h"
#include "arch35/matmul_v3_compile_info_advanced.h"
#include "./arch35/batch_matmul_v3_tiling_advanced.h"
#include "batch_mat_mul_v3_base_tiling.h"

#include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"

using namespace optiling::batch_mat_mul_v3;
using namespace optiling::matmul_v3;

namespace optiling {

REGISTER_TILING_TEMPLATE("BatchMatMulV3", BatchMatmulV3BaseTiling, 0);

static ge::graphStatus BatchMatMulV3TilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("BatchMatMulV3", "context is null"),
                    return ge::GRAPH_FAILED);
    if (IsAdvancedSocVersion(context)) {
        return batch_matmul_v3_advanced::BatchMatMulV3Tiling(context).DoTiling();
    }
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForBatchMatMulV3(gert::TilingParseContext *context) {
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("BatchMatMulV3", "context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
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

    gert::GemmCompileInfo tbeCompileInfo;
    tbeCompileInfo.ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfo);
    tbeCompileInfo.core_num = compileInfoPtr->aicNum;
    OP_TILING_CHECK(tbeCompileInfo.core_num <= 0L,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "aicNum value is [%d]", tbeCompileInfo.core_num),
                    return ge::GRAPH_FAILED);
    optiling::PlatformInfo::GetInstance().SetInstance(tbeCompileInfo);
    OP_LOGI(
        context->GetNodeName(),
        "compile info success soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
        static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->l1Size, compileInfoPtr->l2Size,
        compileInfoPtr->aicNum, compileInfoPtr->supportL0c2out, compileInfoPtr->supportL12BtBf16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchMatMulV3)
    .Tiling(BatchMatMulV3TilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForBatchMatMulV3)
    .GenSimplifiedKey(GenSimplifiedKey);
}