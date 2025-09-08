/* *
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

/* !
 * \file matmul_v3_compile_info_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMPILE_INFO_ADVANCED_H__
#define __OP_HOST_MATMUL_V3_COMPILE_INFO_ADVANCED_H__

#include <string>

#include "../matmul_v3_compile_info.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "mc2_log.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace matmul_v3_advanced {

inline ge::graphStatus InitCompileInfo(fe::PlatFormInfos *platformInfo,
                                       MatmulV3CompileInfo *compileInfoPtr) {
  OP_TILING_CHECK(
      platformInfo == nullptr,
      CUBE_INNER_ERR_REPORT("MatMul", "InitCompileInfo platformInfo is null"),
      return ge::GRAPH_FAILED);
  OP_TILING_CHECK(
      compileInfoPtr == nullptr,
      CUBE_INNER_ERR_REPORT("MatMul", "InitCompileInfo compileInfoPtr is null"),
      return ge::GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
  compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
  compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
  compileInfoPtr->supportL0c2out = false;    // Not used
  compileInfoPtr->supportL12BtBf16 = false;  // Not used
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                 compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,
                                 compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,
                                 compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,
                                 compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,
                                 compileInfoPtr->l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                 compileInfoPtr->l2Size);
  OP_LOGI("MatMul",
          "parse compile info success soc:%d, aicNum:%lu, aivNum:%lu, "
          "ubSize:%lu, l1Size:%lu, l2Size:%lu, l0ASize:%lu, "
          "l0BSize:%lu, "
          "l0CSize:%lu.",
          static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->aicNum,
          compileInfoPtr->aivNum, compileInfoPtr->ubSize,
          compileInfoPtr->l1Size, compileInfoPtr->l2Size,
          compileInfoPtr->l0ASize, compileInfoPtr->l0BSize,
          compileInfoPtr->l0CSize);
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InitCompileInfo(gert::TilingParseContext *context) {
  OP_TILING_CHECK(
      context == nullptr,
      CUBE_INNER_ERR_REPORT("MatMul", "InitCompileInfo context is null"),
      return ge::GRAPH_FAILED);
  fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
  auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
  return InitCompileInfo(platformInfo, compileInfoPtr);
}
}  // namespace matmul_v3_advanced
}  // namespace optiling
#endif  // __OP_HOST_MATMUL_V3_COMPILE_INFO_ADVANCED_H__