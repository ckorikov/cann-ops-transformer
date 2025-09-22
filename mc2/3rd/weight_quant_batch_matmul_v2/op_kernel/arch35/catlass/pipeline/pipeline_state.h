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
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STATE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STATE_H

#include <cstdint>

#include "../utils/device_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Catlass {

namespace detail {
constexpr bool IsPowerOfTwo(int n)
{
    return (n & (n - 1)) == 0;
}
} // namespace detail
template <uint8_t Stages_>
struct PipelineState {
    static constexpr uint8_t Stages = Stages_;

    uint64_t index_ = 0;
    uint64_t count_ = 0;

    DEVICE
    uint64_t index() const
    {
        return index_;
    }

    DEVICE
    uint64_t count() const
    {
        return count_;
    }

    DEVICE
    void operator++()
    {
        static_assert(Stages > 0);

        ++count_;
        if constexpr (detail::IsPowerOfTwo(Stages)) {
            index_ = (index_ + 1) & (Stages - 1);
        } else {
            ++index_;
            if (index_ == Stages) {
                index_ = 0;
            }
        }
    }
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif