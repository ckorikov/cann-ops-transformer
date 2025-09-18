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
 * \file allto_allv_grouped_mat_mul_tiling_A3.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_A3_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_A3_H

#include "allto_allv_grouped_mat_mul_tiling_base.h"

namespace optiling {
class AlltoAllvGmmTilingA3 : public AlltoAllvGmmTilingBase
{
public:
    explicit AlltoAllvGmmTilingA3(gert::TilingContext* context) : AlltoAllvGmmTilingBase(context){};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling
#endif