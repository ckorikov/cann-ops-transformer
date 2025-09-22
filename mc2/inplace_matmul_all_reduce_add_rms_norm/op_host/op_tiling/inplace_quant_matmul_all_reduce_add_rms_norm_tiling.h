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
 * \file inplace_quant_matmul_all_reduce_add_rms_norm_tiling.h
 * \brief
 */
#ifndef _INPLACE_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#define _INPLACE_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#include "../../../matmul_all_reduce_add_rms_norm/op_host/op_tiling/quant_matmul_all_reduce_add_rms_norm_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_0, QuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_1, QuantMatmulAllReduceAddRmsNormTilingData);
} // namespace optiling
#endif // _INPLACE_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_