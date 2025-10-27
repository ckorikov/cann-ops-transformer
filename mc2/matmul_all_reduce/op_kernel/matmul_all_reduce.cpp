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
 * \file matmul_all_reduce.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common.h"
// david非量化
#if defined(__DAV_C310__)
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "arch35/matmul_all_reduce_910_general.h"
#include "arch35/matmul_all_reduce_empty_tensor_k_general.h"
#endif
#if defined(WEIGHT_W4_W8)
#include "./arch35/matmul_all_reduce_weight_quant.h"
#endif

#if defined(WEIGHT_F8) || defined(WEIGHT_W4_W8)
#include "./arch35/matmul_all_reduce_weight_quant_adaptive_split.h"
#include "./arch35/matmul_all_reduce_empty_tensor_k_general.h"
using WeightQuantBatchMatmulV2::Arch35::WeightQuantBatchMatmulV2BasicBlockController;
static constexpr WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512}; // b 转置场景
static constexpr WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256}; // b 非转置场景
static constexpr WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {2, 256}; // solomon
#endif
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_INT8)) ||               \
    (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||          \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) &&   \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))) || \
    (((ORIG_DTYPE_X1 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)) &&      \
     ((ORIG_DTYPE_X2 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)))
#include "arch35/matmul_all_reduce_quant_pertoken.h"
#include "arch35/matmul_all_reduce_quant_pertoken_comm_int8.h"
#include "arch35/matmul_all_reduce_quant.h"
#include "arch35/matmul_all_reduce_quant_comm_int8.h"
#include "arch35/matmul_all_reduce_quant_perblock.h"
#endif
#else
#if __CCE_AICORE__ == 220
#ifdef MC2_QUANT
#include "matmul_all_reduce_quant.h"
#include "matmul_all_reduce_quant_pertoken_comm_int8.h"
#ifdef MC2_QUANT_FP16
#include "matmul_all_reduce_quant_fp16_comm_int8.h"
#else
#include "matmul_all_reduce_quant_bf16_comm_int8.h"
#endif
#else
#include "matmul_all_reduce_empty_tensor_k_general.h"
#ifdef MC2_WEIGHT_QUANT
#include "matmul_all_reduce_weight_quant.h"
#else
#include "matmul_all_reduce_910_general.h"
#endif
#endif
#else
#ifdef __CCE_KT_TEST__
#include "rac_server_stub.h"
#else
#include "arch31/rac_server.h"
#endif
#if (ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8)
#include "arch31/matmul_all_reduce_quant_bmm.h"
// 310p归一化weightNZ非量化
#elif (ORIG_DTYPE_X1 != DT_INT8 && ORIG_DTYPE_X2 != DT_INT8 && FORMAT_X2 != FORMAT_ND)
#include "arch31/matmul_all_reduce_unquant_310.h"
#elif (ORIG_DTYPE_X1 != DT_INT8 && (ORIG_DTYPE_X2 == DT_INT8) && FORMAT_X2 != FORMAT_ND)
#include "arch31/matmul_all_reduce_weight_quant_310.h"
#else
#include "arch31/matmul_all_reduce_310_general.h"
#endif // ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8
#endif // __CCE_AICORE__ == 220
#endif // defined(__DAV_C310__)

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16))
#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(templateClass, opTile, opTail, ...)                   \
    do {                                                                                                            \
        GET_TILING_DATA_MEMBER(QuantMatmulAllReduceTilingData, msg, msg, tilingGM);                                 \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                               \
            GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingData, tilingData, tilingGM);                      \
            templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, int8_t, __VA_ARGS__> matmul;  \
            const QuantMatmulAllReduceTilingData* QuantMatmulAllReduceTiling = &tilingData;                         \
            const QuantBatchMatmulV3TilingData* qBmmV3TilingData = &(QuantMatmulAllReduceTiling->tilematmulTiling); \
            const TCubeTiling* mmTilingTile = &(qBmmV3TilingData->matmulTiling);                                    \
            const QuantBatchMatmulV3TilingData* qBmmV3TilingDataTail =                                              \
                &(QuantMatmulAllReduceTiling->tailmatmulTiling);                                                    \
            const TCubeTiling* mmTilingTail = &(qBmmV3TilingDataTail->matmulTiling);                                \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, mmTilingTile, opTail.mm, mmTilingTail);      \
            matmul.Init(                                                                                            \
                aGM, bGM, biasGM, addGM, dequantGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS, &tilingData, \
                &tPipe);                                                                                            \
            matmul.Process(opTile, opTail);                                                                         \
            tPipe.Destroy();                                                                                        \
        }                                                                                                           \
    } while (0)
#endif // (ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16)

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_FLOAT16))
#define INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(templateClass, ...)                    \
    do {                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingData, tilingData, tilingGM);  \
        templateClass<DTYPE_X1, DTYPE_X2, int32_t, DTYPE_Y, int8_t, __VA_ARGS__> op;        \
        op.Init(aGM, bGM, dequantGM, biasGM, addGM, cGM, workspaceGM, &tilingData, &tPipe); \
        op.InitScale(commQuantScale1GM, commQuantScale2GM);                                 \
        op.Process();                                                                       \
    } while (0)
#endif

#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_X2 == DT_INT8))
#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(templateClass, scaleType, opTile, opTail, ...)     \
    do {                                                                                                             \
        GET_TILING_DATA_MEMBER(QuantMatmulAllReduceTilingData, msg, msg, tilingGM);                                  \
        if (msg.debugMode != static_cast<uint8_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {                                \
            GET_TILING_DATA_WITH_STRUCT(QuantMatmulAllReduceTilingData, tilingData, tilingGM);                       \
            templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, scaleType, DTYPE_Y, int8_t, __VA_ARGS__> matmul; \
            const QuantMatmulAllReduceTilingData* QuantMatmulAllReduceTiling = &tilingData;                          \
            const QuantBatchMatmulV3TilingData* qBmmV3TilingData = &(QuantMatmulAllReduceTiling->tilematmulTiling);  \
            const TCubeTiling* mmTilingTile = &(qBmmV3TilingData->matmulTiling);                                     \
            const QuantBatchMatmulV3TilingData* qBmmV3TilingDataTail =                                               \
                &(QuantMatmulAllReduceTiling->tailmatmulTiling);                                                     \
            const TCubeTiling* mmTilingTail = &(qBmmV3TilingDataTail->matmulTiling);                                 \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, mmTilingTile, opTail.mm, mmTilingTail);       \
            matmul.Init(                                                                                             \
                aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS,   \
                &tilingData, &tPipe);                                                                                \
            matmul.Process(opTile, opTail);                                                                          \
            tPipe.Destroy();                                                                                         \
        }                                                                                                            \
    } while (0)
#endif

namespace MatmulAllReduceImpl {}

using namespace AscendC;
using namespace MatmulAllReduceImpl;

extern "C" __global__ __aicore__ void matmul_all_reduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    if (workspaceGM == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_C310__)
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    // 910非量化
    if (TILING_KEY_IS(11000000000000001100UL)) {
        KERNEL_TASK_TYPE(11000000000000001100UL, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_MC2_910_OP_IMPL(MatmulV3Advanced::MatmulAswKernel, Mc2CoreType::ON_CUBE);
    } else if (TILING_KEY_IS(11000000000000000001UL)) {
        INVOKE_MC2_910_OP_IMPL(MatmulV3Advanced::MatmulAswKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    } else if (TILING_KEY_IS(11000000000000000009UL)) {
        KERNEL_TASK_TYPE(11000000000000000009UL, KERNEL_TYPE_MIX_AIV_1_0);
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    }
#endif
    // 除非量化以外，david的tiling均来源于对应的mmv3算子，在其基础上增加对应偏移，具体参看tiling
#if defined(WEIGHT_W4_W8) || defined(WEIGHT_F8)
#if defined(WEIGHT_W4_W8)
#undef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X1
    // perchannel&pertensor的key计划删除，走新tilingkey，跟随matmul，暂时保留
    if (TILING_KEY_IS(100200UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, QuantType::PER_CHANNEL, false, false);
    } else if (TILING_KEY_IS(101200UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, QuantType::PER_CHANNEL, true, false);
    } else if (TILING_KEY_IS(100210UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, QuantType::PER_CHANNEL, false, false);
    } else if (TILING_KEY_IS(101210UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, QuantType::PER_CHANNEL, true, false);
    } else if (TILING_KEY_IS(100300UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, QuantType::PER_GROUP, false, false);
    } else if (TILING_KEY_IS(100310UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, QuantType::PER_GROUP, false, false);
    } else if (TILING_KEY_IS(101300UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, QuantType::PER_GROUP, true, false);
    } else if (TILING_KEY_IS(101310UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, QuantType::PER_GROUP, true, false);
    } else if (TILING_KEY_IS(100100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, QuantType::PER_TENSOR, false, false);
    } else if (TILING_KEY_IS(100110UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, QuantType::PER_TENSOR, false, false);
    } else if (TILING_KEY_IS(101100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(false, QuantType::PER_TENSOR, true, false);
    } else if (TILING_KEY_IS(101110UL)) {
        INVOKE_MC2_WEIGHT_QUANT_KERNEL(true, QuantType::PER_TENSOR, true, false);
    }
#endif
    if (TILING_KEY_IS(2000030004000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000002100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000002120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020000000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020000000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000011100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000011120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000001100UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001120UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, QuantType::PER_TENSOR, DTYPE_BIAS, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(10000000000000000008UL)) {
        KERNEL_TASK_TYPE(10000000000000000008UL, KERNEL_TYPE_MIX_AIV_1_0);
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    }
#if defined(ORIG_DTYPE_X1) && defined(DT_BF16) && (ORIG_DTYPE_X1 == DT_BF16)
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    if (TILING_KEY_IS(2000030004000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000002140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030004000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000002160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000020000000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020000000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020001000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020002000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000020003000012160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, true, QuantType::PER_CHANNEL, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030003000001140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, false, QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030003000001160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            false, true, QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_3);
    } else if (TILING_KEY_IS(2000030004000011140UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(
            true, false, QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_0);
    } else if (TILING_KEY_IS(2000030004000011160UL)) {
        INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(true, true, QuantType::PER_TENSOR, float, VEC_ANTIQUANT_CONFIG_0);
    }
#endif
#endif

#if defined(DAVID_QUANT_INT8_OUT_FP16)
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, true);
    } else if (TILING_KEY_IS(1000000000000000000)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, false);
    } else if (TILING_KEY_IS(1000000000000000011)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, true);
    } else if (TILING_KEY_IS(1000000000000000010)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, false);
    }

    if (TILING_KEY_IS(1000000000000002000)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, false);
    } else if (TILING_KEY_IS(1000000000000002001)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, true);
    } else if (TILING_KEY_IS(1000000000000002010)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, false);
    } else if (TILING_KEY_IS(1000000000000002011)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, true);
    }
#elif defined(DAVID_QUANT_INT8_OUT_BF16)
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(1000000000000000000)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1000000000000000011)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(1000000000000000010)) {
        INVOKE_MC2_QUANT_COMM_INT8_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    }

    if (TILING_KEY_IS(1000000000000002000)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1000000000000002001)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(1000000000000002010)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1000000000000002011)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, DTYPE_Y, false, true);
    }
#elif (                                                                         \
    ((ORIG_DTYPE_X1 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)) && \
    ((ORIG_DTYPE_X2 == DT_FLOAT4_E1M2) || (ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)))
// fp8,hif8和mixfp的场景，bias都是float
#undef DTYPE_BIAS
#define DTYPE_BIAS float
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    }
#elif (                                                                            \
    ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||        \
    (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
     ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
// fp8,hif8和mixfp的场景，bias都是float
#undef DTYPE_BIAS
#define DTYPE_BIAS float
#if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
    if (TILING_KEY_IS(1000000000000000011)) {
        INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    }
#endif
    if (TILING_KEY_IS(1000000000000000001)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, true);
    } else if (TILING_KEY_IS(1000000000000000000)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(AscendC::MatMulASWKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, uint64_t, false, false);
    } else if (TILING_KEY_IS(1000000000000002000)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, false);
    } else if (TILING_KEY_IS(1000000000000002001)) {
        INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(
            QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR, float, false, true);
    } else if (TILING_KEY_IS(1000000000000004000)) {
        INVOKE_MC2_QUANT_PERBLOCK_910_OP_IMPL(
            QuantBatchMatmulV3::MatMulPerBlockASW, Mc2CoreType::ON_CUBE_AND_VECTOR, false, false);
    } else if (TILING_KEY_IS(1000000000000004001)) {
        INVOKE_MC2_QUANT_PERBLOCK_910_OP_IMPL(
            QuantBatchMatmulV3::MatMulPerBlockASW, Mc2CoreType::ON_CUBE_AND_VECTOR, false, true);
    }
#endif
#else
#if __CCE_AICORE__ == 220
#if defined(MC2_QUANT_FP16)
    if (TILING_KEY_IS(1)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequant, Mc2CoreType::ON_CUBE_AND_VECTOR, REG_NO_MM_OBJ, false, int32_t, uint64_t, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(0)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequant, Mc2CoreType::ON_CUBE_AND_VECTOR, REG_NO_MM_OBJ, false, int32_t, uint64_t, DTYPE_Y, false,
            false);
    } else if (TILING_KEY_IS(16)) {
        KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, float, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(17)) {
        KERNEL_TASK_TYPE(17, KERNEL_TYPE_MIX_AIC_1_1);
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, float, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(11)) {
        INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(MatmulAllReduceQuantFP16CommInt8, false, true);
    } else if (TILING_KEY_IS(10)) {
        INVOKE_QUANT_BMM_DEQUANT_FP16_IMPL_COMM_INT8(MatmulAllReduceQuantFP16CommInt8, false, false);
    } else if (TILING_KEY_IS(26)) { // pertoken 适配 int8 通信
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, false, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, false, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(
            MatmulAllReduceQuantPertokenInt8, float, opTile, opTail, false, false);
    } else if (TILING_KEY_IS(27)) { // pertoken 适配 int8 通信
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, true, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, float, DTYPE_Y, false, true, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(
            MatmulAllReduceQuantPertokenInt8, float, opTile, opTail, false, true);
    }
#elif defined(MC2_QUANT_BF16)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if (TILING_KEY_IS(0)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequantBf16, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, false, DTYPE_Y, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequantBf16, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, false, DTYPE_Y, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(16)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, DTYPE_Y, DTYPE_Y, false, false);
    } else if (TILING_KEY_IS(17)) {
        INVOKE_MC2_QUANT_910_OP_IMPL(
            BmmDequantPertoken, Mc2CoreType::ON_VECTOR, REG_MM_OBJ, true, DTYPE_Y, DTYPE_Y, false, true);
    } else if (TILING_KEY_IS(10)) {
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_2);
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTile;
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(
            MatmulAllReduceQuantBF16CommInt8, opTile, opTail, false, false);
    } else if (TILING_KEY_IS(11)) {
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTile;
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8(
            MatmulAllReduceQuantBF16CommInt8, opTile, opTail, false, true);
    } else if (TILING_KEY_IS(26)) {
        KERNEL_TASK_TYPE(26, KERNEL_TYPE_MIX_AIC_1_2);
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, false, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(
            MatmulAllReduceQuantPertokenInt8, DTYPE_Y, opTile, opTail, false, false);
    } else if (TILING_KEY_IS(27)) {
        KERNEL_TASK_TYPE(27, KERNEL_TYPE_MIX_AIC_1_2);
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTile;
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_Y, DTYPE_Y, false, true, true> opTail;
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_COMM_INT8_IMPL(
            MatmulAllReduceQuantPertokenInt8, DTYPE_Y, opTile, opTail, false, true);
    }
#elif defined(MC2_WEIGHT_QUANT)
    if (TILING_KEY_IS(365056114230017)) {   // 310100
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, QuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(365330992136961)) { // 311100
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, QuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(365056651100929)) { // 310110
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, QuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(365331529007873)) { // 311110
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, QuantType::PER_TENSOR, true);
#if (FORMAT_X2 == FORMAT_FRACTAL_NZ)
    } else if (TILING_KEY_IS(365057188299521)) { // 810200
#else
    } else if (TILING_KEY_IS(365057187971841)) { // 310200
#endif
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, QuantType::PER_CHANNEL, false);
#if (FORMAT_X2 == FORMAT_FRACTAL_NZ)
    } else if (TILING_KEY_IS(365332066206465)) { // 811200
#else
    } else if (TILING_KEY_IS(365332065878785)) { // 311200
#endif
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, QuantType::PER_CHANNEL, true);
#if (FORMAT_X2 == FORMAT_FRACTAL_NZ)
    } else if (TILING_KEY_IS(365057725170433)) { // 810210
#else
    } else if (TILING_KEY_IS(365057724842753)) { // 310210
#endif
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, QuantType::PER_CHANNEL, false);
#if (FORMAT_X2 == FORMAT_FRACTAL_NZ)
    } else if (TILING_KEY_IS(365332603077377)) { // 811210
#else
    } else if (TILING_KEY_IS(365332602749697)) { // 311210
#endif
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, QuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(365058261713665)) { // 310300
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, QuantType::PER_GROUP, false);
    } else if (TILING_KEY_IS(365058798584577)) { // 310310
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, QuantType::PER_GROUP, false);
    } else if (TILING_KEY_IS(365333139620609)) { // 311300
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(false, QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(365333676491521)) { // 311310UL
        INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(true, QuantType::PER_GROUP, true);
    } else if (TILING_KEY_IS(10000000000000000008UL)) { // 空kernel模板
        KERNEL_TASK_TYPE(10000000000000000008UL, KERNEL_TYPE_MIX_AIV_1_0);
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    }
#else
    if (TILING_KEY_IS(10000000000000001100UL)) {
        KERNEL_TASK_TYPE(10000000000000001100UL, KERNEL_TYPE_MIX_AIC_1_0);
        INVOKE_MC2_910_OP_IMPL(MatmulBaseKernel, Mc2CoreType::ON_CUBE);
    } else if (TILING_KEY_IS(65536UL)) {
        INVOKE_MC2_910_OP_IMPL(MatmulBaseKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    } else if (TILING_KEY_IS(0UL)) {
        INVOKE_MC2_910_OP_IMPL(MatmulBaseUnAlignedKernel, Mc2CoreType::ON_CUBE_AND_VECTOR);
    } else if (TILING_KEY_IS(10000000000000000009UL)) {
        KERNEL_TASK_TYPE(10000000000000000009UL, KERNEL_TYPE_MIX_AIV_1_0);
        INVOKE_MC2_EMPTY_TENSOR_OP_IMPL();
    }
#endif // ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_Y == DT_FLOAT16
#else
    // 310P
    HcclServer hcclServer;
#if (ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8) // 归一化310p weightNZ全量化
    if (TILING_KEY_IS(1U)) {
        INVOKE_QUANT_BMM_OP_IMPL(MatmulAllReduceQuantBmm, false, true);
    }
#elif (ORIG_DTYPE_X1 == DT_FLOAT16 && ORIG_DTYPE_X2 == DT_INT8 && FORMAT_X2 != FORMAT_ND) // 归一化310p weightNZ伪量化
    // transA, transB, antiQuantType, hasAntiQuantOffset
    if (TILING_KEY_IS(13195213800193)) { // 80010
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, QuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(13195482235649)) {   // 80011
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, QuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(13196287542017)) {   // 80020
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, QuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(13196555977473)) {   // 80021
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, QuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(13470091707137)) {   // 80110
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, QuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(13470360142593)) {   // 80111
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, QuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(13471165448961)) {   // 80120
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, false, true, QuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(13471433884417)) {   // 80121
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_310(MatmulAllReduceWeightQuant310, true, true, QuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(2100000UL)) { // 空kernel
        GET_TILING_DATA_WITH_STRUCT(WeightQuantMatmulAllReduceNzTilingData, tilingData, tilingGM);
        WeightQuantEmptyTensorKernel(biasGM, cGM, workspaceGM, &tilingData, &hcclServer);
    }
#elif (ORIG_DTYPE_X1 != DT_INT8 && ORIG_DTYPE_X2 != DT_INT8 && FORMAT_X2 != FORMAT_ND) // 310p归一化weightNZ非量化
    if (TILING_KEY_IS(2000UL)) {
        INVOKE_UNQUANT_BMM_OP_IMPL_310(MatmulAllReduceUnquant310);
    } else if (TILING_KEY_IS(67536UL)) {
        INVOKE_UNQUANT_BMM_OP_IMPL_310(MatmulAllReduceUnquant310);
    } else if (TILING_KEY_IS(2100000UL)) {
        // k==0 伪量化NZ
        GET_TILING_DATA_WITH_STRUCT(UnQuantMatmulAllReduceTilingData, tilingData, tilingGM);
        MatMulEmptyTensorKernelUnquantNz(biasGM, cGM, workspaceGM, &tilingData, &hcclServer);
    }
#else  // 310p weightND(非归一化 非量化 伪量化)
    // L2cache, transB, isWeightQuant, AntiQuantType, hasAntiQuantOffset, 310Version
    if (TILING_KEY_IS(2000001UL)) {
        // 开启L2cache, 非量化ND，非量化-不涉及antiQuantType，非量化-不涉及antiQuantOffset
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, false, AntiQuantType::NONE, false);
    } else if (TILING_KEY_IS(2000000UL)) {
        // 关闭L2cache, 非量化ND，非量化-不涉及antiQuantType，非量化-不涉及antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, false, AntiQuantType::NONE, false);
    } else if (TILING_KEY_IS(2001110UL)) {
        // 关闭L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(2001111UL)) {
        // 打开L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(2002110UL)) {
        // 关闭L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(2002111UL)) {
        // 打开L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(2011110UL)) {
        // 关闭L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(2011111UL)) {
        // 打开L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(2012110UL)) {
        // 关闭L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(2012111UL)) {
        // 打开L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, true>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(2001100UL)) {
        // 关闭L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(2001101UL)) {
        // 打开L2cache, 伪量化ND，perTensor，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, false);
    } else if (TILING_KEY_IS(2002100UL)) {
        // 关闭L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(2002101UL)) {
        // 打开L2cache, 伪量化ND，perChannel，无antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, false);
    } else if (TILING_KEY_IS(2011100UL)) {
        // 关闭L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(2011101UL)) {
        // 打开L2cache, 伪量化ND，perTensor，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_TENSOR, true);
    } else if (TILING_KEY_IS(2012100UL)) {
        // 关闭L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, false, true, AntiQuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(2012101UL)) {
        // 打开L2cache, 伪量化ND，perChannel，带antiQuantOffset，weight非转置
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, true>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        INVOKE_MATMUL_ALL_REDUCE_OP_IMPL(MatmulAllReduce310General, true, true, AntiQuantType::PER_CHANNEL, true);
    } else if (TILING_KEY_IS(2100000UL)) {
        // k==0 伪量化ND
        GET_TILING_DATA(tilingData, tilingGM);
        MatMulEmptyTensorKernel(biasGM, cGM, workspaceGM, &tilingData, &hcclServer);
    }
#endif // ORIG_DTYPE_X1 == DT_INT8
#endif // __CCE_AICORE__ == 220 or not
#endif // __DAV_C310__
}
