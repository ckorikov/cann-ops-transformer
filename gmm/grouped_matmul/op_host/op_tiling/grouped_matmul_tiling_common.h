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
 * \file grouped_matmul_tiling_common.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_COMMON_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_COMMON_H

// #include "torch_extension/tiling_utils.h"
#include "platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include "grouped_matmul_tiling_temp.h"
class GroupedMatmulTiling {
public:
    template <typename T1, typename T2, typename T3>
    static void GroupedMatmulCommonTiling(T1 x, T1 weight, T2 bias, T2 scale, T2 offset, T2 antiquantScale,
                                          T2 antiquantOffset, T3 groupList, T2 perTokenScale, GMMTilingData tilingData,
                                          uint32_t coreNum, uint64_t ubSize)
    {
    }
};

#endif
