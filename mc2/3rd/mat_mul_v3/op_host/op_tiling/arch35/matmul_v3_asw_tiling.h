/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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


/* !
 * \file matmul_v3_asw_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ASW_TILING_H__
#define __OP_HOST_MATMUL_V3_ASW_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3AswTiling : public MatMulV3BaseTiling {
public:
    MatMulV3AswTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3BaseTiling(context, cfg) {};

    ~MatMulV3AswTiling() override {};

protected:
    bool IsCapable() override
    {
        return true;
    };

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    MatMulV3Model aswtModel_ {MatMulV3Model::BASIC};

private:
    void FormulateBasicBlock();
    void CalcTailBasicBlock();
    void OptimizeEdgeBasicBlock();
    uint64_t GetOuterAxisTailCnt(const uint64_t x, const uint64_t y, const uint64_t baseX,
                                 const uint64_t baseY, const uint64_t aicNum) const;
};
} // namespace matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ASW_TILING_H__