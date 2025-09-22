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
 * \file weight_quant_batch_matmul_v2_tiling_tool.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H

#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"
#include "mc2_log.h"

using AscendC::BLOCK_CUBE;
using AscendC::ONE_BLK_SIZE;
using matmul_tiling::MatrixTraverse;

namespace optiling {

constexpr uint64_t BASIC_BLOCK = 512UL;

template <typename T1, typename T2>
T2 CalcTailSize(T1 num1, T2 num2)
{
    if (num2 == 0) {
        return 0;
    }

    T1 mod = num1 % static_cast<T1>(num2);
    return mod != 0 ? static_cast<T2>(mod) : num2;
}

int64_t GetDtypeBits(ge::DataType dtype);

uint64_t GetBlockAlignSizeByDataType(ge::DataType dtype);

uint64_t GetShapeSizeWithDataType(uint64_t shapeSize, ge::DataType dtype);

bool CheckOptionalInputByShape(const gert::StorageShape* storageShape);

matmul_tiling::DataType GetMatmulTilingDtype(ge::DataType dtype);

ge::Format GetInputStorageFormat(const gert::TilingContext* context, size_t id);
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TOOL_H