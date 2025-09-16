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
 * \file moe_distribute_dispatch_tiling_a2a3.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_A2A3_H
#define MOE_DISTRIBUTE_DISPATCH_TILING_A2A3_H

#include "tiling/moe_tiling_base.h"
#include "moe_distribute_dispatch_tiling_helper.h"

namespace optiling {
class MoeDistributeDispatchTilingA2A3 : public MoeTilingBase {
public:
    explicit MoeDistributeDispatchTilingA2A3(gert::TilingContext *context) : MoeTilingBase (context) {};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_DISPATCH_TILING_A2A3_H