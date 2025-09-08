/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file mat_mul_pingpong_basic_act.h
 * \brief
 */
#ifndef MAT_MUL_PINGPONG_BASIC_ACT_H
#define MAT_MUL_PINGPONG_BASIC_ACT_H
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "block_scheduler_aswt.h"
#include "include/epilogue/block_epilogue_empty.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul_without_que.h"
using namespace Act;
using namespace Act::Gemm;
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void MatMulActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, const MatMulV3BasicTilingData& tilingData, int64_t batch = 0)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;

    if (batch > 1 && biasGM != nullptr) {
        return;
    }

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = BuiltInAswtScheduler<FULL_LOAD_MODE>;

    // 定义MMAD类型
    using BlockMmad = Block::BlockMmadBuilder<
            AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC,
            L1TileShape, L0TileShape, BlockScheduler, MatmulMultiBlockWithOutQue<>>;

    // 定义Fusion类型
    using FusionOp = Block::DefaultFusion<OutType, OutType>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义Kernel类型
    using MatmulKernel = Kernel::KernelMatmulWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
        {aGM, bGM, cGM, biasGM}, // gm addr
        {}, // epilogue args
        {&tilingData}
    };
    AscendC::TPipe tPipe;
    MatmulKernel mm;
    mm(params);
}
#endif