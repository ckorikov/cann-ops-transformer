/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_BLOCK_BLOCK_DECL_H
#define ARCH35_CATLASS_BLOCK_BLOCK_DECL_H

#include "lib/std/type_traits.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
template <
    typename DispatchPolicy, typename TileShapeL1, typename TileShapeL0, typename DtypeA, typename StrideA,
    typename DtypeB, typename StrideB, typename DtypeBias, typename StrideBias, typename DtypeC, typename StrideC>
struct BlockMmad {
    static_assert(!AscendC::Std::is_same_v<DispatchPolicy, DispatchPolicy>, "Unsupported combination");
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif