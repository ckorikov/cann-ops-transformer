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
 * \file matmul_v3_tiling_strategy.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_STRATEGY_H__
#define __OP_HOST_MATMUL_V3_STRATEGY_H__

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace matmul_v3_advanced {
namespace strategy {
constexpr int32_t STREAM_K = 0;
constexpr int32_t FULL_LOAD_BASE = 2;
constexpr int32_t BASIC_ASWT = 1;
constexpr int32_t BASE = 999;

const static std::map<platform_ascendc::SocVersion, std::vector<int32_t>> MatMulV3PrioritiesMap = {
    { platform_ascendc::SocVersion::ASCEND910_95,
    { strategy::STREAM_K, strategy::BASIC_ASWT, strategy::FULL_LOAD_BASE} },
};

inline std::vector<int32_t> GetMatMulV3Priorities(platform_ascendc::SocVersion socVersion)
{
    std::vector<int32_t> priorities = {};
    if (MatMulV3PrioritiesMap.find(socVersion) != MatMulV3PrioritiesMap.end()) {
        priorities = MatMulV3PrioritiesMap.at(socVersion);
    }
    return priorities;
};
}
}
}

#endif // __OP_HOST_MATMUL_V3_STRATEGY_H__
