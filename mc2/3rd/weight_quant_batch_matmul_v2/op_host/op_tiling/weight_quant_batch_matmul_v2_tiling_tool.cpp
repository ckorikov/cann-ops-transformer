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
 * \file weight_quant_batch_matmul_v2_tiling_tool.cpp
 * \brief
 */
#include "weight_quant_batch_matmul_v2_tiling_tool.h"

namespace optiling {

constexpr int64_t B32_BITS = 32;
constexpr int64_t B16_BITS = 16;
constexpr int64_t B8_BITS = 8;
constexpr int64_t B4_BITS = 4;

uint64_t GetBlockAlignSizeByDataType(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return ONE_BLK_SIZE + ONE_BLK_SIZE;
    } else {
        return ONE_BLK_SIZE / static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t GetShapeSizeWithDataType(uint64_t shapeSize, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4) {
        return (shapeSize + 1) >> 1;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

bool CheckOptionalInputByShape(const gert::StorageShape* storageShape)
{
    return storageShape != nullptr && storageShape->GetStorageShape().GetShapeSize() != 0;
}

int64_t GetDtypeBits(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return B4_BITS;
    } else if (
        dtype == ge::DT_INT8 || dtype == ge::DT_HIFLOAT8 || dtype == ge::DT_FLOAT8_E5M2 ||
        dtype == ge::DT_FLOAT8_E4M3FN) {
        return B8_BITS;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return B16_BITS;
    } else if (dtype == ge::DT_FLOAT) {
        return B32_BITS;
    } else {
        return 0;
    }
}

const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP = {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_FLOAT8_E8M0, matmul_tiling::DataType::DT_FLOAT8_E8M0},
    {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
    {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_FLOAT4_E2M1},
    {ge::DT_FLOAT4_E1M2, matmul_tiling::DataType::DT_FLOAT4_E1M2},
};

matmul_tiling::DataType GetMatmulTilingDtype(ge::DataType dtype)
{
    auto it = DTYPE_MAP.find(dtype);
    // impossible to get runtime error
    return it != DTYPE_MAP.end() ? it->second : matmul_tiling::DataType::DT_FLOAT16;
}

ge::Format GetInputStorageFormat(const gert::TilingContext* context, size_t id)
{
    auto desc = context->GetInputDesc(id);
    OP_TILING_CHECK(
        desc == nullptr,
        OP_LOGE("weight_quant_batch_matmul_v2_tiling", "get input[%zu] Desc is null!", id),
        return ge::FORMAT_NULL);
    return static_cast<ge::Format>(GetPrimaryFormat(desc->GetStorageFormat()));
}
} // namespace optiling