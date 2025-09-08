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
 * \file matmul_v3_tiling.cc
 * \brief
 */
#include "matmul_v3_tiling.h"

#include <type_traits>

#include "op_cache_tiling.h"
#include "arch35/matmul_v3_compile_info_advanced.h"
#include "arch35/matmul_v3_tiling_advanced.h"
#include "matmul_v3_base_tiling.h"
#include "matmul_v3_compile_info.h"
#include "matmul_v3_simplifiedkey.h"
#include "matmul_v3_platform_common.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace optiling::matmul_v3;
using Ops::NN::Optiling::TilingRegistry;

namespace {
static const size_t DEST_MAX = 100;
static const size_t MAX_LEN_SIMPLIFIED_KEY = 256;
static const int32_t INPUT0_INDEX = 0;
static const int32_t INPUT1_INDEX = 1;
static const int32_t BIAS_INDEX = 2;
}

namespace optiling {

REGISTER_TILING_TEMPLATE("MatMulV3", MatmulV3BaseTiling, 0);

static ge::graphStatus MatmulV3TilingFunc(gert::TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"), return ge::GRAPH_FAILED);
  if (IsAdvancedSocVersion(context)) {
      return matmul_v3_advanced::MatMulV3Tiling(context).DoTiling();
  }
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMatmulV3(gert::TilingParseContext *context) {
  if (IsAdvancedSocVersion(context)) {
      return matmul_v3_advanced::InitCompileInfo(context);
  }
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"), return ge::GRAPH_FAILED);
  fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
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
  compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;                       // 1024 is btSize
  compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize;  // 4096 is btSize
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

  if(!TilingPrepareForOpCache(context)) {
      return ge::GRAPH_FAILED;
  }
  OP_LOGI(
      context->GetNodeName(),
      "parse compile info success soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
      static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->l1Size, compileInfoPtr->l2Size,
      compileInfoPtr->aicNum, compileInfoPtr->supportL0c2out, compileInfoPtr->supportL12BtBf16);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatMulV3)
    .Tiling(MatmulV3TilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForMatmulV3)
    .GenSimplifiedKey(GenSimplifiedKey);
}
