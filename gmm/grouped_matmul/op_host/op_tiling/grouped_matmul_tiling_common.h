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

#include "grouped_matmul_tiling.h"
class GroupedMatmulTiling {
    template <typename T>
    static void GroupedMatmulCommonTiling(T x, T weight, T bias, T scale, T offset, T antiquantScale, T antiquantOffset,
                                          T groupList, T perTokenScale, GMMTilingData tilingData, uint32_t coreNum,
                                          uint64_t ubSize)
    {
    }
}

#endif
