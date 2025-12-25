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
namespace GroupedMatmulNs {
class GroupedMatmulTiling {
public:
    // 使用更灵活的模板
    template <typename TensorListType, typename OptionalTensorListType, typename OptionalTensorType>
    static void GroupedMatmulCommonTiling(
        const TensorListType& x, 
        const TensorListType& weight,
        const OptionalTensorListType& bias,
        const OptionalTensorListType& scale,
        const OptionalTensorListType& offset,
        const OptionalTensorListType& antiquantScale,
        const OptionalTensorListType& antiquantOffset,
        const OptionalTensorType& groupList,
        const OptionalTensorListType& perTokenScale,
        GMMTilingData& tilingData,  // 改为引用
        uint32_t coreNum, 
        uint64_t ubSize)
    {
        // 实现
    }
};
}
#endif
