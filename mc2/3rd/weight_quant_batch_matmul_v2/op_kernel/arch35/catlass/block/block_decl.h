/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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