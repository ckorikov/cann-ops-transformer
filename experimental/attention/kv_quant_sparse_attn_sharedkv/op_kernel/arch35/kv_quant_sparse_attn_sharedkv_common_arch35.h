/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_flash_attention_comm.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_AHSREDKV_COMMON_ARCH35_H
#define KV_QUANT_SPARSE_ATTN_AHSREDKV_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../kv_quant_sparse_attn_sharedkv_common.h"

constexpr static int64_t SPARSE_MODE_INT_DEFAULT = 2147483647;

constexpr uint64_t BLOCK_BYTE = 32;
constexpr int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t pse1NS1S2 = 2;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;
// 0级接口的block间隔范围需要满足32B对齐
constexpr int64_t attenMaskBN2GS1S2 = 0;
constexpr int64_t attenMaskBS1S2 = 1;
constexpr int64_t attenMaskS1S2 = 2;
constexpr int64_t attenMaskTT = 99;
constexpr uint16_t PREFIX_N_MAX_B = 32;
constexpr int32_t fp32BaseSize = 8;

constexpr uint32_t attenMaskNoCompress = 0;
constexpr uint32_t attenMaskLeftUpCausalCompress = 1;
constexpr uint32_t attenMaskRightDownCausalCompress = 2;
constexpr uint16_t SHIFT_NUM_2 = 2;
constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint32_t L0C_SHARED_SIZE_64K = 64 * 1024;
constexpr uint32_t L0C_SHARED_SIZE_128K = 128 * 1024;
#if (__NPU_ARCH__ == 5102)
constexpr uint32_t CV_RATIO = 1;
#else
constexpr uint32_t CV_RATIO = 2;
#endif
constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8}; // mm2ResIntraEvent
constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10}; //mm1ResIntraEvent
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint64_t MLA_L0A_SIZE = 64;
constexpr uint64_t MLA_L0B_SIZE = 64; 
constexpr uint64_t BASE_SIZE_128 = 128;
constexpr uint64_t FLOAT_BYTES = 4;

namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

static constexpr uint32_t FA_BYTE_BLOCK = 32;

__aicore__ constexpr uint16_t Align64Func(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}
}

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, SAS_LAYOUT LAYOUT_T, \
    SAS_KV_LAYOUT KV_LAYOUT_T, SASTemplateMode TEMPLATE_MODE>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, SAS_LAYOUT, SAS_LAYOUT::BSND) \
    X(KV_LAYOUT_T, SAS_KV_LAYOUT, SAS_KV_LAYOUT::PA_ND) \
    X(TEMPLATE_MODE, SASTemplateMode, SASTemplateMode::SCFA_TEMPLATE_MODE) \


/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif
