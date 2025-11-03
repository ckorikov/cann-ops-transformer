/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_reduce_scatter_tiling.cpp
 * \brief
 */
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include <graph/utils/type_utils.h>
#include <register/op_def_registry.h>
#include <platform/platform_infos_def.h>

#include "tiling/mc2_tiling_utils.h"
#include "mc2_log.h"

using namespace AscendC;
using namespace ge;

namespace {

} // namespace

namespace optiling {

static ge::graphStatus QuantReduceScatterTilingFunc(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    return GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantReduceScatter(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantReduceScatter)
    .Tiling(QuantReduceScatterTilingFunc);
} // namespace optiling
