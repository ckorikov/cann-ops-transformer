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
 * \file rotary_position_embedding_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/rotary_position_embedding_bab.h"
#include "arch35/rotary_position_embedding_ab.h"
#include "arch35/rotary_position_embedding_aba_and_ba.h"
#include "arch35/rotary_position_embedding_a_and_b.h"

#define TILING_KEY_ABA 20010
#define TILING_KEY_BA 20011
#define TILING_KEY_BAB 20020
#define TILING_KEY_AB  20030
#define TILING_KEY_A 20040
#define TILING_KEY_B 20041

using namespace AscendC;
using namespace RotaryPositionEmbedding;

extern "C" __global__ __aicore__ void rotary_position_embedding(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_ABA)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingABAAndBA<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingABAAndBA<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BAB)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingBAB<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_AB)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseABTilingData, tiling_data_in, tiling);
        const RopeRegbaseABTilingData *__restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingAB<DTYPE_X> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_A)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingAAndB<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingAAndB<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    }
}
