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
#ifndef ARCH35_CATLASS_UTILS_CONSTANT_H
#define ARCH35_CATLASS_UTILS_CONSTANT_H

#include "lib/std/type_traits.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
using _0 = AscendC::Std::integral_constant<uint32_t, 0>;
using _1 = AscendC::Std::integral_constant<uint32_t, 1>;
// constant 16
using _16 = AscendC::Std::integral_constant<uint32_t, 16>;
// constant 32
using _32 = AscendC::Std::integral_constant<uint32_t, 32>;
// constant 64
using _64 = AscendC::Std::integral_constant<uint32_t, 64>;
// constant 256
using _256 = AscendC::Std::integral_constant<uint32_t, 256>;
// constant 512
using _512 = AscendC::Std::integral_constant<uint32_t, 512>;
// constant 1024
using _1024 = AscendC::Std::integral_constant<uint32_t, 1024>;
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif