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
 * \file matmul_tiling_cfg.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_TILING_CFG_H__
#define __OP_HOST_MATMUL_TILING_CFG_H__

#include <cstdint>
#include <cstddef>
#include <vector>

namespace optiling {
struct TilingResult {
    uint64_t tilingKey;
    uint64_t blockDim;
    void *tilingData;
    size_t tilingDataSize;
    std::vector<size_t> workspaceSize;
};

class MatMulTilingCfg {
public:
    MatMulTilingCfg(bool needUpdateIn, const void *compileInfoIn, const void *argsIn)
        : needUpdate(needUpdateIn), compileInfo(compileInfoIn), args(argsIn)
    {}

    virtual ~MatMulTilingCfg() {};

    virtual void Update(const TilingResult &result) {
        (void)result;
    };

public:
    const bool needUpdate = false;     // true: Tiling结果通过Update返回； false: Tiling结果填充到context中
    const void *compileInfo = nullptr; // 编译信息，用于生成TilingKey
    const void *args = nullptr;        // 算子参数，用于生成TilingKey
};
} // namespace optiling

#endif // __OP_HOST_MATMUL_TILING_CFG_H__
