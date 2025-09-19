/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
 * \file rope_rotate_half.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_ROTATE_HALF_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_ROTATE_HALF_H
#include "register/op_def_registry.h"
#include "rotary_position_embedding_tiling.h"

namespace optiling {

class RopeRotateHalfTilingClass : public RotaryPosEmbeddingMembaseTilingClass {
public:
    explicit RopeRotateHalfTilingClass(gert::TilingContext *context) : RotaryPosEmbeddingMembaseTilingClass(context)
    {
    }

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        if (socVersion_ == platform_ascendc::SocVersion::ASCEND910B && inputMode_ != MODE_ROTATE_INTERLEAVED) {
            return true;
        }
        return false;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_ROTATE_HALF_H
