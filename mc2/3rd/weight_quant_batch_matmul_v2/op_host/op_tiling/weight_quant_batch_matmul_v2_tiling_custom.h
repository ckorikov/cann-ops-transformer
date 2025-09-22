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
 * \file weight_quant_batch_matmul_v2_tiling_custom.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {
class WeightQuantBatchMatmulV2TilingCustom : public WeightQuantBatchMatmulV2Tiling
{
public:
    explicit WeightQuantBatchMatmulV2TilingCustom(gert::TilingContext* context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    WeightQuantBatchMatmulV2TilingCustom(gert::TilingContext* context, WeightQuantBatchMatmulV2TilingData* out)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
        tilingData_ = out;
        InitCompileInfo();
        isOutTilingData_ = true;
    }
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~WeightQuantBatchMatmulV2TilingCustom() override = default;

protected:
    WeightQuantBatchMatmulV2TilingData* tilingData_ = nullptr;
    std::unique_ptr<WeightQuantBatchMatmulV2TilingData> tilingDataManager_;
    // mc2信息
    bool isOutTilingData_ = false;
    uint64_t cubeBaseN_;

    bool IsCapable() override;
    void Reset();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    bool GetMatMulTiling();
    void SetShapeSize();
    void AdjustMatmulTiling() const;
    void AdjustL1Size() const;
    void ComputeDefaultBlock(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    void ComputeGroupDefaultBlock(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    void ReviseGroupDefaultBlockWithTrans(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    void ReviseGroupDefaultBlockWithoutTrans(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    void ComputeVectorDefaultBlock(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    void ComputeInt4VectorDefaultBlock(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    uint64_t ComputeAntiquantBuffer(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN);
    uint64_t ComputeWeightBuffer(uint64_t defaultVecSingleK, uint64_t defaultVecSingleN);
    void ComputeInt8VectorDefaultBlock(uint64_t& defaultVecSingleK, uint64_t& defaultVecSingleN) const;
    bool GetTilingFromCache();
    bool CheckCacheTiling();
    bool InvokeCacheTiling();
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_H
