/**
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

/*!
 * \file matmul_v3_l2_cache.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_L2_CACHE_H__
#define __OP_HOST_MATMUL_V3_L2_CACHE_H__

#include "matmul_v3_tiling.h"
#include "matmul_v3_common.h"

namespace optiling {
namespace matmul_v3 {
class L2Cache {
public:
    L2Cache(MatmulV3Args &args, MatmulTilingData &tilingData)
        : args_(args), tilingData_(tilingData) {
    }
    void SetL2CacheFlag(TilingEnable tilingEnable, uint64_t l2Size, uint32_t &l2CacheFlag);
private:
    void SetL2CacheFlag(bool aEnableL2Cache, bool bEnableL2Cache, bool cEnableL2Cache,
                        bool biasEnableL2Cache, uint32_t &l2CacheFlag);
    void SetL2CacheFlagMultiCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
    void SetL2CacheFlagSingleCoreSplitK(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
    void SetL2CacheFlagBase(bool &aEnableL2Cache, bool &bEnableL2Cache) const;
private:
    MatmulV3Args &args_;
    MatmulTilingData &tilingData_;
};
}
}
#endif // __OP_HOST_MATMUL_V3_L2_CACHE_H__