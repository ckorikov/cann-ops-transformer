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
 * \file matmul_v3_asw_basic_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_basic_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3AswBasicApiTiling, ASCEND910_95, BASIC_ASWT);

bool MatMulV3AswBasicApiTiling::IsCapable()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_256);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256);
    if (mCore * nCore > compileInfo_.aicNum){
        OP_LOGD(args_.opName, "mCnt_[%lu] and nCnt_[%lu] is not enter in matmulv3 basic api", mCore, nCore);
        return false;
    }
    if (args_.bFormat != ge::FORMAT_ND || args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for basic api");
        return false;
    }
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state basic api");
    return true;
}


uint64_t MatMulV3AswBasicApiTiling::GetTilingKey() const
{
    return MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}
} // namespace matmul_v3
} // namespace optiling