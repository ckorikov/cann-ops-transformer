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
 * \file matmul_all_reduce_tiling_310_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
#define MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
#include "../matmul_all_reduce_tiling.h"
namespace optiling {
class MatmulAllReduceTiling310General : public MatmulAllReduceTilingBase
{
public:
    explicit MatmulAllReduceTiling310General(gert::TilingContext* context) : MatmulAllReduceTilingBase(context)
    {}
    ~MatmulAllReduceTiling310General() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    void DoMatmulTiling310(
        matmul_tiling::MultiCoreMatmulTiling& mm1, TCubeTiling& cubeTiling, L2cacheTilePara& l2cacheTiling);

    void DoWeightAntiQuantTiling();

    void GetL2CacheParm(
        uint64_t& l2CacheSize, uint64_t& singleMatrixSize, uint32_t& tileSize, uint32_t& tileLimit, bool useNewPara);

    void SetTransLength(matmul_tiling::MultiCoreMatmulTiling& mm1, TCubeTiling& cubeTiling);

private:
    bool isTransB_ = false;
    bool isWeightQuant_ = false;
    AntiQuantType antiQuantT_ = AntiQuantType::NONE;
    bool hasAntiQuantOffset_ = false;
};
} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
