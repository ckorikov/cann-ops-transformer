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
#ifndef ARCH35_CATLASS_UTILS_MATH_UTILS_H
#define ARCH35_CATLASS_UTILS_MATH_UTILS_H

#include "device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
template <typename T>
DEVICE T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
DEVICE T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { X_LOG("Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
DEVICE T Min(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? a : b;
#else
    return min(a, b);
#endif
}

template <typename T>
DEVICE T Max(T a, T b)
{
#if defined(__CCE_KT_TEST__)
    return a < b ? b : a;
#else
    return max(a, b);
#endif
}
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif