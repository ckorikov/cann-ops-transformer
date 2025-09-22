/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file weight_quant_batch_matmul_v2_compute_matmul_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "op_cache_tiling.h"

using Ops::Transformer::OpTiling::AiCoreParams;

namespace optiling {
struct MatmulMultiCoreResult {
    uint8_t mDim;
    uint8_t nDim;
    uint8_t batchDim;
};

struct MatmulParams {
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType cDtype;
    ge::DataType biasDtype;
    bool transA;
    bool transB;
    bool hasBias;
    ge::Format format_a;
    ge::Format format_b;
    ge::Format format_out;
    QuantType quantType;
    bool kbAlign;
};

class ComputeMatmulTiling
{
public:
    static bool GetTiling(
        TCubeTiling& matmulTiling, MatmulMultiCoreResult& multiCoreResult, const MatmulParams& params,
        const AiCoreParams& aicoreParams, gert::TilingContext* context);

private:
    static bool GetCacheTiling(
        TCubeTiling& matmulTiling, MatmulMultiCoreResult& multiCoreResult, const MatmulParams& params,
        gert::TilingContext* context);

    static void CalcCommonTiling(
        TCubeTiling& matmulTiling, const MatmulParams& params, const AiCoreParams& aicoreParams);

    static void CalcMsdBufferSize(TCubeTiling& matmulTiling, const MatmulParams& params);

    static bool MsdA16W8CommonTiling(
        TCubeTiling& matmulTiling, MatmulMultiCoreResult& multiCoreResult, const MatmulParams& params,
        const AiCoreParams& aicoreParams);

    static bool SimpleIncreTiling(
        TCubeTiling& matmulTiling, MatmulMultiCoreResult& multiCoreResult, const MatmulParams& params,
        const AiCoreParams& aicoreParams);

    static void Convert2AscendCTiling(
        const CacheTilingData& tbeTiling, TCubeTiling& matmulTiling, const MatmulParams& params,
        MatmulMultiCoreResult& multiCoreResult);
    static MatrixTraverse GetIteratorOrder(
        const CacheTilingData& tbeTiling, int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK,
        ge::DataType aDtype);

    static bool tryComputeSimpleTiling(
        TCubeTiling& matmulTiling, const MatmulParams& params, const AiCoreParams& aicoreParams);

    static bool tryAFullLoad(TCubeTiling& matmulTiling, const MatmulParams& params, const AiCoreParams& aicoreParams);

    static bool trySimpleTilingNormalLoad(
        TCubeTiling& matmulTiling, const MatmulParams& params, const AiCoreParams& aicoreParams);
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H
