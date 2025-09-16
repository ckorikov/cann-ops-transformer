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
 * \file moe_distribute_combine_tiling_helper.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H
#define MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "mc2_log.h"
#include "tiling/tiling_base.h"

namespace optiling {
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t EXPAND_IDX_INDEX = 2;
constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
constexpr uint32_t EXPERT_SCALES_INDEX = 4;
constexpr uint32_t TP_SEND_COUNTS_INDEX = 5;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 6;
constexpr uint32_t OUTPUT_X_INDEX = 0;

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;

class MoeDistributeCombineTilingHelper {
public:
    static ge::graphStatus TilingCheckMoeDistributeCombine(gert::TilingContext *context, const char *nodeName);

protected:
    static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName);
    static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName);
    static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName);

private:
    inline static bool CheckInputTensorDim(gert::TilingContext *context, const char *nodeName);
    inline static bool CheckInputSendCountsTensorDim(gert::TilingContext *context, const char *nodeName);
    inline static bool CheckInputExpertScalesTensorDim(gert::TilingContext *context, const char *nodeName);
    inline static bool CheckOutputTensorDim(gert::TilingContext *context, const char *nodeName);
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H