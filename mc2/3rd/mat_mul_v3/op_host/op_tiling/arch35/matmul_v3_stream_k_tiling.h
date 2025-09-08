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
 * \file matmul_v3_stream_k_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__
#define __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3StreamKTiling : public MatMulV3BaseTiling {
public:
    MatMulV3StreamKTiling(gert::TilingContext *context, MatMulTilingCfg &cfg) : MatMulV3BaseTiling(context, cfg) {}

    ~MatMulV3StreamKTiling() override {}

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    std::vector<size_t> GetWorkspaceSize() const override;

private:
    bool CheckStreamKSKTiling() const;
    bool CheckStreamKDPSKTiling() const;
    MatMulV3L0C2Out GetL0C2OutFlag() const;

    uint64_t mCnt_{ 1 };
    uint64_t nCnt_{ 1 };
    uint64_t totalMNCnt_{ 1 };
    MatMulV3L0C2Out l0C2Out_{MatMulV3L0C2Out::ON_THE_FLY};
};
}
}

#endif // __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__
