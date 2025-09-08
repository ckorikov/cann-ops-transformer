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
 * \file matmul_v3_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
#define __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__

#include "runtime/tiling_context.h"
#include "matmul_v3_common_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3Tiling {
public:
    explicit MatMulV3Tiling(gert::TilingContext *context) : context_(context){};
    virtual ~MatMulV3Tiling() = default;
    virtual ge::graphStatus DoTiling();

protected:
    virtual ge::graphStatus GetShapeAttrsInfo();

    virtual ge::graphStatus GetArgs();

    virtual ge::graphStatus CheckArgs();

protected:
    gert::TilingContext *context_ = nullptr;
    MatMulV3Args args_;
};
}
}
#endif // __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
