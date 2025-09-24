/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_utils.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#if defined(V310_GMM_QUANT)
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_init_output.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h"
#endif
#if defined(V310_GMM_QUANT_PERTILE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_act_pertile_kernel.h"
#endif
#elif defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/basic_block_config.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/weight_quant_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h"
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::A16MXF4_NZKN;
using WeightQuantBatchMatmulV2::Arch35::S8S4_NZKN_G;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantVcvMatmulBasicBlock;
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {3, 512};
#if defined(DT_FLOAT) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_FLOAT
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT fp4x2_e2m1_t
#endif
#else
#include "arch35/non_quant/grouped_matmul_basic_kernel.h"
#endif
#else
#include "grouped_matmul_antiquant.h"
#include "grouped_matmul_vector.h"
#include "grouped_matmul.h"
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220

#include "grouped_matmul_antiquant_a16w8_msd.h"
#include "grouped_matmul_antiquant_a8w4_msd_pre.h"
#include "grouped_matmul_antiquant_a8w4_msd.h"
#include "grouped_matmul_antiquant_a8w4_pre.h"
#include "grouped_matmul_antiquant_a8w4.h"
#include "grouped_matmul_antiquant_a8w4_msd_new.h"
#include "grouped_matmul_quant_mixcore.h"
#include "grouped_matmul_pre_tiling.h"
#include "grouped_matmul_a4w4.h"
#include "grouped_matmul_autotiling_a8w4.h"
#endif


using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
constexpr MatmulConfig matmulCFG = NZ_CFG_MDL;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
constexpr MatmulConfig matmulCFG = CFG_MDL;
#endif

#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
constexpr auto GetMmCFG() {
    auto CFG = CFG_MDL;
    CFG.isPartialOutput = true;
    return CFG;
}
constexpr MatmulConfig A8W4_GMM_CFG_MDL_NEW = GetMmCFG();
#endif
}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using xTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

template <bool trans = false>
using weightTypeMSD = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_WEIGHT, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using yTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

namespace {
    __aicore__ inline static constexpr MatmulApiStaticTiling GetGmmMatmulApiTiling(bool isND2NZ, bool transB) {
        MatmulConfig conf = GenGmmConf(isND2NZ);
        MatmulApiStaticTiling staticTilingTmp;
        if (transB) {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<true>, yType, biasType>(conf);
        } else {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<false>, yType, biasType>(conf);
        }
        staticTilingTmp.depthA1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.depthB1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.stepM = 1;
        staticTilingTmp.stepN = 1;
        staticTilingTmp.stepKa = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.stepKb = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.dbL0A = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0B = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0C = 1;
        return staticTilingTmp;
    }
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
    constexpr bool isWeightNZ = true;
#else
    constexpr bool isWeightNZ = false;
#endif
    constexpr static auto staticCFG = GetGmmMatmulApiTiling(isWeightNZ, false);
    constexpr static auto staticCFGtransB = GetGmmMatmulApiTiling(isWeightNZ, true);
} // namespace


#define GMM_IMP(computeClass, processClass, transA, transB, sync, cfg)                                             \
    do {                                                                                                           \
        using matmulType = MMType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                        \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);                                       \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_STATIC_TILING_IMP(processClass, transA, transB, sync, cfg)                                        \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init((TCubeTiling*)nullptr, &tPipe);                                                                    \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, nullptr, &tPipe);                                               \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, nullptr, 0, groupList, tiling);                                                   \
        op.InitStaticTiling((cfg).baseM, (cfg).baseN);                                                             \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_STATIC_TILING_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType) \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init((TCubeTiling*)nullptr, &tPipe);                                                                \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, nullptr, &tPipe);                                                \
        computeOp.InitStaticTiling(&gmmBaseParams_, user1, (cfg).baseM, (cfg).baseN);                              \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, nullptr, 0, groupList, tiling);                                                   \
        op.InitStaticTiling((cfg).baseM, (cfg).baseN);                                                             \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                                                      \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, &mmTilingData_, &tPipe);                                        \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#if defined(CONST_TILING)
#define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)               \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GMMTilingData gmmTilingData;                                                                               \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)
#else
    #define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)           \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        GMMPreTilingProcess preTiling;                                                                             \
        preTiling.Init(groupList, gmmBaseParams_, mmTilingData_, &tPipe);                                          \
        preTiling.Process(gmmBaseParams_, mmTilingData_);                                                          \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)
#endif

#define GMM_A4W4_IMP(computeClass, transA, transB, cfg, aType, bType, cType)                                       \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> computeOp(mm);                                                                    \
        computeOp.Init(x, weight, scale, groupList, perTokenScale,                                                 \
                    y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                            \
        computeOp.Process();                                                                                       \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, cfg)                                                               \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4PreProcess op1;                                                                                 \
            op1.Init(x, x, groupList, user1, gmmBaseParams_, &tPipe);                                              \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD, false>;                          \
        using bT = MatmulType<TPosition::GM, wFormat, DTYPE_WEIGHT_DEV_A8W4MSD, false>;                            \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                         \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>;                                                     \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> op(mm);                                                                           \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, offset, nullptr, nullptr, nullptr,               \
                y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4(computeClass, cfg)                                                                   \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4FakeQuantPreProcess<wFormat> op1;                                                               \
            op1.Init(weight, y, groupList, user1, gmmBaseParams_, &tPipe);                                         \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        SyncAll<false>();                                                                                          \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;                                       \
        using bT = MatmulType<TPosition::GM, wFormat, int8_t, false>;                                              \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                         \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>;                                                     \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> op(mm);                                                                           \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, offset, nullptr, nullptr, nullptr,               \
                    y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                            \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_FAKEA8W8(computeClass, cfg)                                                          \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4FakeQuantPreProcess<wFormat> op1;                                                               \
            op1.Init(weight, y, scale, user1, gmmBaseParams_, &tPipe);                                             \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        SyncAll<false>();                                                                                          \
        GlobalTensor<int8_t> yGm;                                                                                  \
        yGm.SetGlobalBuffer((__gm__ int8_t *)workspace);                                                           \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;                                       \
        using bT = MatmulType<TPosition::GM, wFormat, int8_t, false>;                                              \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                      \
        using matmulType = MMImplType<aT, bT, cT, biasT, matmulCFG>;                                               \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        GMMQuantMixCoreCompute<matmulType, false> computeOp(mm);                                                   \
        computeOp.isA8W4FakeQuant = true;                                                                          \
        computeOp.Init(x, user1, bias, user1, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,   \
                       y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                         \
        GMMProcess<decltype(computeOp)> op(computeOp);                                                             \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_CONTROLLER_OP_IMPL(templateClass, ...)                                           \
    do {                                                                                                         \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);           \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,            \
                      WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                                              \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(templateClass, ...)                                             \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,                  \
                      WeightQuantVcvMatmulBasicBlock, __VA_ARGS__> op;                                                 \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                                  \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat,                                \
                      transposeX1, transposeX2>                                                                        \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_MIX_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                              \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, wFormat,                         \
                      transposeX1, transposeX2, DTYPE_L0C_LOCAL>                                                       \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        if ASCEND_IS_AIC {                                                                                             \
            templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, DTYPE_Y, wFormat,                            \
                          transposeX1, transposeX2>                                                                    \
                op;                                                                                                    \
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,      \
                    gmmArrayAddr_, &tPipe);                                                                            \
            op.Process();                                                                                              \
        }                                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            GQmmEmptyTensor<DTYPE_Y>(groupList, y, &gmmQuantParams_, gmmArrayAddr_, mmTilingData_.usedCoreNum,         \
                                     &tPipe);                                                                          \
        }                                                                                                              \
    } while (0)

#define GMM_QUANT_GB_IMPL_CLASS(xLayout, wLayout, yLayout)                                                             \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GmmActPerTileKernel<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, yLayout, \
                            DTYPE_L0C_LOCAL>(x, weight, bias, scale, groupList, perTokenScale, y, user1,               \
                                             &gmmQuantParams_, &mmTilingData_, &tPipe);                                \
    } while (0)

extern "C" __global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_QUANT) // Quant: A8W8
#if defined(V310_GMM_QUANT_MX) // mxfpx
    if (TILING_KEY_IS(20000000000)) { // transX = false, transW = false, groupType = 0
        KERNEL_TASK_TYPE(20000000000, KERNEL_TYPE_AIC_ONLY);
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if (TILING_KEY_IS(20000000001)) { // transX = false, transW = true, groupType = 0
        KERNEL_TASK_TYPE(20000000001, KERNEL_TYPE_AIC_ONLY);
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // scale64/perTensor/double perTensor
    if (TILING_KEY_IS(20000000000)) { // transX = false, transW = false, groupType = 0
        KERNEL_TASK_TYPE(20000000000, KERNEL_TYPE_AIC_ONLY);
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if (TILING_KEY_IS(20000000001)) { // transX = false, transW = true, groupType = 0
        KERNEL_TASK_TYPE(220000000001, KERNEL_TYPE_AIC_ONLY);
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // mx/perTensor/double perTensor
    if (TILING_KEY_IS(20000000010)) { // transX = true, transW = false, groupType = 2
        KERNEL_TASK_TYPE(20000000010, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(true, false, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MIX) // perToken/SPLIT_K/scale bf16/fp32
    if (TILING_KEY_IS(20000000100)) { // transX = false, transW = false, groupType = 0
        KERNEL_TASK_TYPE(20000000100, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_MIX_IMPL_CLASS(false, false, GQmmMixRegbaseKernel);
    } else if (TILING_KEY_IS(20000000101)) { // transX = false, transW = true, groupType = 0
        KERNEL_TASK_TYPE(20000000101, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_MIX_IMPL_CLASS(false, true, GQmmMixRegbaseKernel);
    } else if (TILING_KEY_IS(20000000110)) { // transX = true, transW = false, groupType = 2
        KERNEL_TASK_TYPE(20000000110, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_MIX_IMPL_CLASS(true, false, GQmmMixRegbaseKernel);
    }
#endif
#if defined(V310_GMM_QUANT_PERTILE)
    if (TILING_KEY_IS(20000000200)) { // transX = false, transW = false, groupType = 0
        KERNEL_TASK_TYPE(20000000200, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajor,
                                Act::Gemm::layout::RowMajorAlign);
    } else if (TILING_KEY_IS(20000000201)) { // transX = false, transW = true, groupType = 0
        KERNEL_TASK_TYPE(20000000201, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::RowMajor, Act::Gemm::layout::ColumnMajor,
                                Act::Gemm::layout::RowMajorAlign);
    } else if (TILING_KEY_IS(20000000210)) { // transX = true, transW = false, groupType = 2
        KERNEL_TASK_TYPE(20000000210, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajor,
                                Act::Gemm::layout::RowMajorAlign);
    }
#endif
#elif defined(V310_GMM_ANTI_QUANT)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    #if ORIG_DTYPE_X == DT_INT8
        if (TILING_KEY_IS(2000020004000002001UL)) {
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, S8S4_NZKN_G,
                                                           VEC_ANTIQUANT_CONFIG_4);
        }
    #elif ORIG_DTYPE_ANTIQUANT_SCALE == DT_FLOAT8_E8M0
        if (TILING_KEY_IS(2000020003000004001UL)) {
            INVOKE_GMM_WEIGHT_QUANT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, A16MXF4_NZKN,
                                                       VEC_ANTIQUANT_CONFIG_3);
        }
    #else
        if (TILING_KEY_IS(2000020003000012000UL)) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,
                                                QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                       VEC_ANTIQUANT_CONFIG_3);
        } else if (TILING_KEY_IS(2000020003000012020UL)) {
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,
                                                QuantType::NONE, CubeFormat::ND};
            INVOKE_GMM_WEIGHT_QUANT_CONTROLLER_OP_IMPL(GMMWeightQuantResplitController, wqmmCfg,
                                                       VEC_ANTIQUANT_CONFIG_3);
        }
    #endif
#else
    if (TILING_KEY_IS(10000900009000090000UL)) {
        if constexpr (wFormat == CubeFormat::NZ) {
            GmmNoQuantAswt<layout::RowMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);
        } else {
            GmmNoQuantAswt<layout::RowMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);
        }
    } else if (TILING_KEY_IS(10000900009000090001UL)) {    // x transposed
        KERNEL_TASK_TYPE(10000900009000090001UL, KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIV {
            EmptyTensor<DTYPE_Y>(groupList, y, tiling);
        }
        if ASCEND_IS_AIC {
            if constexpr (wFormat == CubeFormat::NZ) {
                GmmNoQuantAswt<layout::ColumnMajor, layout::Nz>(x, weight, bias, groupList, y, tiling);
            } else {
                GmmNoQuantAswt<layout::ColumnMajor, layout::RowMajor>(x, weight, bias, groupList, y, tiling);
            }
        }
    } else if (TILING_KEY_IS(10000900009000090002UL)) {    // weight transposed
        if constexpr (wFormat == CubeFormat::NZ) {
            GmmNoQuantAswt<layout::RowMajor, layout::Zn>(x, weight, bias, groupList, y, tiling);
        } else {
            GmmNoQuantAswt<layout::RowMajor, layout::ColumnMajor>(x, weight, bias, groupList, y, tiling);
        }
    }
#endif
#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#if defined(GMM_ANTI_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(8)) {  // antiquant msd
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDCompute, A8W4_GMM_CFG_MDL);
    } else if (TILING_KEY_IS(12)) {  // antiquant msd
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDComputeNew, A8W4_GMM_CFG_MDL_NEW);
    } else if (TILING_KEY_IS(17)) {  // antiquant per channel
        KERNEL_TASK_TYPE(17, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP_A8W4_FAKEA8W8(GMMA8W4Compute, A8W4_GMM_CFG_MDL);
    } else if (TILING_KEY_IS(18)) {  // antiquant per group
        KERNEL_TASK_TYPE(18, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP_A8W4(GMMA8W4Compute, A8W4_GMM_CFG_MDL);
    } else if (TILING_KEY_IS(21)) {
        KERNEL_TASK_TYPE(21, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_MEMBER(GMMTilingData, hpTilingData, tilingData, tiling);
        GM_ADDR A = x;
        GM_ADDR B = weight;
        GM_ADDR C = y;
        GM_ADDR groupListOptional = groupList;
        GM_ADDR bias_ = bias;
        GM_ADDR offset_ = offset;
        GM_ADDR sa = perTokenScale;
        GM_ADDR sw = scale;
        GM_ADDR workspaceDevice = user1;

        GMMA4W8AutotilingCompute op(A, B, C, groupListOptional, bias_, offset_, sa, sw, workspaceDevice,
                                    const_cast<A8W4HPTiling *>(&tilingData), &tPipe);
        op.Init();
        op.Process();
    }
#elif defined(GMM_ANTI_QUANT)
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, false, false, matmulCFG);
    } else if (TILING_KEY_IS(2)) {  // weight tansposed
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, true, false, matmulCFG);
    } else if (TILING_KEY_IS(3)) {  // antiquant performence
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_IMP(GMMAntiquantComputePerformance, GMMAntiquantProcess, false, false, false, matmulCFG);
    }
    #if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT8) && ORIG_DTYPE_WEIGHT == DT_INT8
        if (TILING_KEY_IS(6)) {  // antiquant msd
            KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIC_1_1);
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, false, false, matmulCFG,
                            xTypeMSD, weightTypeMSD, yTypeMSD);
        } else if (TILING_KEY_IS(7)) {  // antiquant msd weight tansposed
            KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_1);
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, true, false, matmulCFG,
                            xTypeMSD, weightTypeMSD, yTypeMSD);
        }
    #endif
#elif defined(GMM_QUANT_BF16) || defined(GMM_QUANT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(2)) {  // weight tansposed
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(5)) {  // weight tansposed
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(9)) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess, false, false, false, matmulCFG, xType,
                         weightType, yType);
    } else if (TILING_KEY_IS(10)) {  // weight tansposed
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess, false, true, false, matmulCFG, xType,
                         weightType, yType);
    } else if (TILING_KEY_IS(17)) {
        KERNEL_TASK_TYPE(17, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMProcess,
                                       false, false, false, staticCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(18)) {
        KERNEL_TASK_TYPE(18, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMProcess,
                                       false, true, false, staticCFGtransB, xType, weightType, yType);
    } else if (TILING_KEY_IS(19)) {
        KERNEL_TASK_TYPE(19, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess,
                                       false, false, false, staticCFG, xType, weightType, yType);
    } else if (TILING_KEY_IS(20)) {
        KERNEL_TASK_TYPE(20, KERNEL_TYPE_MIX_AIC_1_1);
        GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess,
                                       false, true, false, staticCFGtransB, xType, weightType, yType);
    }
#elif defined(GMM_A4W4)
    if(TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        GMM_A4W4_IMP(GMMA4W4Compute, false, false, matmulCFG, xType, weightType, yType);
    }
#else
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
    } else if (TILING_KEY_IS(2)) {    // weight transposed
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
    }
    #if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_X == ORIG_DTYPE_WEIGHT && \
        ORIG_DTYPE_X == DT_INT8
    if (TILING_KEY_IS(9)) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_IMP(GMMGroupMSparseProcess, false, false, false, matmulCFGUnitFlag);
    } else if (TILING_KEY_IS(10)) {    // weight transposed
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_IMP(GMMGroupMSparseProcess, false, true, false, matmulCFGUnitFlag);
    } else if (TILING_KEY_IS(13)) {
        KERNEL_TASK_TYPE(13, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_STATIC_TILING_IMP(GMMProcess, false, false, false, staticCFG);
    } else if (TILING_KEY_IS(14)) {
        KERNEL_TASK_TYPE(14, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_STATIC_TILING_IMP(GMMProcess, false, true, false, staticCFGtransB);
    } else if (TILING_KEY_IS(15)) {
        KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_STATIC_TILING_IMP(GMMGroupMSparseProcess, false, false, false, staticCFG);
    } else if (TILING_KEY_IS(16)) {
        KERNEL_TASK_TYPE(16, KERNEL_TYPE_MIX_AIC_1_0);
        GMM_CUBE_STATIC_TILING_IMP(GMMGroupMSparseProcess, false, true, false, staticCFGtransB);
    }
    #endif
#endif

#if defined(GMM_FLOAT)
    if (TILING_KEY_IS(1)) {    // x transposed
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);
            EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
        }
        if ASCEND_IS_AIC {
            GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
        }
    }
#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
#if defined(GMM_FLOAT)
    if (TILING_KEY_IS(0)) {
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFG);
    } else if (TILING_KEY_IS(1)) {    // x transposed
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);
            EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
        }
        if ASCEND_IS_AIC {
            GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
        }
    } else if (TILING_KEY_IS(2)) {    // weight transposed
        GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFG);
    }

#endif
#endif
}
