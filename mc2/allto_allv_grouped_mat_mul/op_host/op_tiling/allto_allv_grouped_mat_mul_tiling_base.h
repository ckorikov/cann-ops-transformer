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
 * \file allto_allv_grouped_mat_mul_tiling_base.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling/mc2_tiling_struct.h"
#include "tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "tiling/mc2_tiling_utils.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {

class AlltoAllvGmmTilingBase : public TilingBaseClass
{
public:
    explicit AlltoAllvGmmTilingBase(gert::TilingContext* context) : TilingBaseClass(context){};

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    platform_ascendc::SocVersion socVersion_;
};
} // namespace optiling

#endif // MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_H
