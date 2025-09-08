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
 * \file matmul_v3_asw_basic_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__
#define __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"
#include "matmul_v3_asw_tiling.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3AswBasicApiTiling : public MatMulV3AswTiling {
public:
    MatMulV3AswBasicApiTiling(gert::TilingContext *context, MatMulTilingCfg &cfg)
        : MatMulV3AswTiling(context, cfg) {};

    ~MatMulV3AswBasicApiTiling() override {};

protected:
    bool IsCapable() override;

    uint64_t GetTilingKey() const override;
};
} // namespace matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__