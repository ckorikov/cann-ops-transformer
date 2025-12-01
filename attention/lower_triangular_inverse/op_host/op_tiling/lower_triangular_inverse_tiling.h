/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_inverse_tiling.h
 * \brief
 */
#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_INVERSE_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_INVERSE_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "err/ops_err.h"

namespace optiling {

struct LowerTriangularInverseCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t btSize{0UL};
    float cubeFreq{0};
    platform_ascendc::SocVersion socVersion;
    bool supportL0c2out = false;
    bool supportL12BtBf16 = false;
};

BEGIN_TILING_DATA_DEF(LowerTriangularInverseTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
  TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LowerTriangularInverse, LowerTriangularInverseTilingData)
REGISTER_TILING_DATA_CLASS(LowerTriangularInverseTilingDataOp, LowerTriangularInverseTilingData)
}
#endif // __OP_HOST_CHUNK_GATED_DELTA_RULE_INVERSE_H__