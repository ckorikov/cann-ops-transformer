/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file matmul_v3_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_H__
#define __OP_HOST_MATMUL_V3_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(L2cacheUseInfo)
  TILING_DATA_FIELD_DEF(uint32_t, l2CacheFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheUseInfoOp, L2cacheUseInfo);

BEGIN_TILING_DATA_DEF(L2cacheTilePara)
  TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);
  TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);
  TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, calOrder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheTileParaOp, L2cacheTilePara)

BEGIN_TILING_DATA_DEF(MatMulRunInfo)
  TILING_DATA_FIELD_DEF(uint32_t, transA);
  TILING_DATA_FIELD_DEF(uint32_t, transB);
  TILING_DATA_FIELD_DEF(uint32_t, nd2nzA);
  TILING_DATA_FIELD_DEF(uint32_t, nd2nzB);
  TILING_DATA_FIELD_DEF(uint32_t, isNzA);
  TILING_DATA_FIELD_DEF(uint32_t, isNzB);
  TILING_DATA_FIELD_DEF(uint32_t, isHf32);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulRunInfoOp, MatMulRunInfo)

BEGIN_TILING_DATA_DEF(MatmulTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF_STRUCT(L2cacheTilePara, tileL2cacheTiling);
  TILING_DATA_FIELD_DEF_STRUCT(MatMulRunInfo, matmulRunInfo);
  TILING_DATA_FIELD_DEF_STRUCT(L2cacheUseInfo, l2cacheUseInfo);
  TILING_DATA_FIELD_DEF(uint32_t, baseAN);
  TILING_DATA_FIELD_DEF(uint32_t, baseAD);
  TILING_DATA_FIELD_DEF(uint32_t, baseBN);
  TILING_DATA_FIELD_DEF(uint32_t, baseBD);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulV3, MatmulTilingData)
REGISTER_TILING_DATA_CLASS(MatmulTilingDataOp, MatmulTilingData)
}
#endif // __OP_HOST_MATMUL_V3_TILING_H__