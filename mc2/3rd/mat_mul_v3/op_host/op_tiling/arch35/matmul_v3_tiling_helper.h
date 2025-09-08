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
 * \file matmul_v3_tiling_helper.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_HELPER_H__
#define __OP_HOST_MATMUL_V3_TILING_HELPER_H__

#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_key.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3TilingHelper {
public:
    static void ResetBase(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo);
    static void CalL1Tiling(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo);
    static MatMulV3L0C2Out GetL0C2Out(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
        const MatMulV3RunInfo &runInfo);
    static bool CheckIfDoubleAswt(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                  const uint64_t batchC);
};
}
}

#endif // __OP_HOST_MATMUL_V3_TILING_HELPER_H__
