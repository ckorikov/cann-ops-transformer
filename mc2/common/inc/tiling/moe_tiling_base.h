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
 * \file moe_tiling_base.h
 * \brief
 */

#ifndef MOE_TILING_BASE_H
#define MOE_TILING_BASE_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "mc2_log.h"
#include "tiling_base/tiling_base.h"
#include "tiling/mc2_tiling_struct.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "tiling/mc2_tiling_utils.h"
#include "platform/platform_infos_def.h"

namespace optiling {

class MoeTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeTilingBase(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};
protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    platform_ascendc::SocVersion socVersion_;
};
} // namespace optiling

#endif // MOE_TILING_BASE_H