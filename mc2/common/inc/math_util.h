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
 * \file math_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_

#include "util/math_util.h"

namespace Ops {
namespace NN {
class MathUtil {
 public:
  template <typename T>
  static T CeilDivision(T num1, T num2) {
    return Ops::Base::CeilDiv(num1, num2);
  }

  static int64_t CeilDivision(int64_t num1, int32_t num2) {
    return Ops::Base::CeilDiv(num1, static_cast<int64_t>(num2));
  }
  template <typename T>
  static T Align(T num1, T num2) {
    return Ops::Base::CeilAlign(num1, num2);
  }
};
}  // namespace NN
}  // namespace Ops

namespace ops {
template <typename T>
static T CeilAlign(T num1, T num2) {
  return Ops::Base::CeilAlign(num1, num2);
}
template <typename T>
static T FloorAlign(T num1, T num2) {
  return Ops::Base::FloorAlign(num1, num2);
}
template <typename T>
static T FloorDiv(T num1, T num2) {
  return Ops::Base::FloorDiv(num1, num2);
}
template <typename T>
static T CeilDiv(T num1, T num2) {
  return Ops::Base::CeilDiv(num1, num2);
}
}  // namespace ops

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_MATH_UTIL_H_
