/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tbe_tiling_api.h"

using namespace optiling;

namespace optiling {
bool GetTbeTiling(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoForV2, Conv3dBackpropV2TBETilingData& tbeTilingForV2)
{
    return true;
}
bool GetTbeTiling(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
    Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType)
{
    return true;
}
}