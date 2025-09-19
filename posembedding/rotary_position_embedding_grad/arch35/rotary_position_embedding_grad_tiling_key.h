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
 * \file rotary_position_embedding_grad_tiling_key.h
 * \brief rotary_position_embedding_grad key
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_TILING_KEY_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_TILING_KEY_H__

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define ROPE_GRAD_BIT_WIDTH 8

ASCENDC_TPL_ARGS_DECL(
    RotaryPositionEmbeddingGrad, REDUCE_TPL_KEY_DECL(),
    ASCENDC_TPL_UINT_DECL(DxTilingKey, ROPE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
    ASCENDC_TPL_UINT_DECL(DcosFlag, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    // Empty
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_EMPTY(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    // A
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_A(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    // ARA
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 201, 202, 203, 204, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_GROUP(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 201, 202, 203, 204, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    // ARARARARA  RARA模版，Reduce会自动补齐维到ARARARARA
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARARARARA_GROUP(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)));

#endif