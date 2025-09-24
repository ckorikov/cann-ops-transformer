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

/*!
 * \file tiling_key.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace Ops {
namespace Transformer {
namespace Optiling {
constexpr uint64_t RecursiveSum() { return 0; }
constexpr uint64_t BASE_MULTIPLIER_SCALE = 10;

template <typename T, typename... Args>
constexpr uint64_t RecursiveSum(T templatedId, Args... templatedIds) {
  return static_cast<uint64_t>(templatedId) +
         BASE_MULTIPLIER_SCALE * RecursiveSum(templatedIds...);
}

constexpr uint64_t TILINGKEYOFFSET = uint64_t(10000000000000000000UL);
template <typename... Args>
constexpr uint64_t GET_TILINGKEY(Args... templatedIds) {
  return TILINGKEYOFFSET + RecursiveSum(templatedIds...);
}

#ifndef TILINGKEY
#define TILINGKEY(ub2, ub1, block, dtype, layout, sparse)       \
  (GER_TILINGKEY(AxisEnum::ub2, AxisEnum::ub1, AxisEnum::block, \
                 DtypeEnum::dtype, LayoutEnum::layout, SparseEnum::sparse))
#endif
}  // namespace Optiling
}  // namespace Transformer
}  // namespace Ops