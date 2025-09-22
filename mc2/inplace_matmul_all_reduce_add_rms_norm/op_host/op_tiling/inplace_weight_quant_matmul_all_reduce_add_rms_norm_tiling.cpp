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
 * \file inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.cc
 * \brief
 */
#ifndef _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#define _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_

#include "inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.h"
namespace optiling {
namespace {
constexpr char MRN[] = "MatmulAllReduceAddRmsNorm";
constexpr char IMRN[] = "InplaceMatmulAllReduceAddRmsNorm";
} // namespace
using InplaceWeightQuantMatmulAllReduceAddRmsNormTiling = WeightQuantMatmulAllReduceAddRmsNormTiling;
REGISTER_TILING_TEMPLATE(IMRN, InplaceWeightQuantMatmulAllReduceAddRmsNormTiling, 1);
} // namespace optiling
#endif // _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_