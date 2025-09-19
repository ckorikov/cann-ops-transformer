/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file rope_half_grad.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_HALF_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_HALF_GRAD_H

#include "rotary_position_embedding_grad_tiling.h"

namespace optiling {

class RopeRotateHalfGradTlingClass : public RotaryPosEmbeddingGradMembaseTilingClass
{
public:
    explicit RopeRotateHalfGradTlingClass(gert::TilingContext* context)
        : RotaryPosEmbeddingGradMembaseTilingClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_HALF_GRAD_H
