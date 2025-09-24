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
 * \file inplace_matmul_all_reduce_add_rms_norm_tiling.cpp
 * \brief
 */
#ifndef _INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#define _INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#include "inplace_matmul_all_reduce_add_rms_norm_tiling.h"

using Ops::Transformer::OpTiling::TilingRegistry;
namespace optiling {
namespace {
constexpr char MRN[] = "MatmulAllReduceAddRmsNorm";
constexpr char IMRN[] = "InplaceMatmulAllReduceAddRmsNorm";
} // namespace

struct DefaultCompileInfo {
};
static ge::graphStatus DefaultTilingParseFunc(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
};
static ge::graphStatus DefaultTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}
IMPL_OP_OPTILING(InplaceMatmulAllReduceAddRmsNorm)
    .Tiling(DefaultTilingFunc)
    .TilingParse<DefaultCompileInfo>(DefaultTilingParseFunc);

using InplaceMatmulAllReduceAddRmsNormTiling = MatmulAllReduceAddRmsNormTiling;
REGISTER_TILING_TEMPLATE(IMRN, InplaceMatmulAllReduceAddRmsNormTiling, 2);
} // namespace optiling
#endif // _INPLACE_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_