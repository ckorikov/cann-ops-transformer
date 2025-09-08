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
 * \file matmul_v3_platform_common.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__
#define __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__

#include "exe_graph/runtime/tiling_parse_context.h"
namespace optiling {
const std::initializer_list<platform_ascendc::SocVersion> AdvancedSocVersion = {
    platform_ascendc::SocVersion::ASCEND910_95};

template <typename T>
inline typename std::enable_if<
    std::is_same<T, gert::TilingParseContext>::value || std::is_same<T, gert::TilingContext>::value, bool>::type
IsAdvancedSocVersion(T *context) {
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platform_ascendc::SocVersion socVersion = ascendcPlatform.GetSocVersion();
    return std::find(AdvancedSocVersion.begin(), AdvancedSocVersion.end(), socVersion) != AdvancedSocVersion.end();
}
}
#endif // __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__
