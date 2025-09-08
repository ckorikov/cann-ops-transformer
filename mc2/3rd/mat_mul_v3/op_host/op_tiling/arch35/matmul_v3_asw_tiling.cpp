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
 * \file matmul_v3_asw_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3AswTiling, ASCEND910_95, BASE);

void MatMulV3AswTiling::CalcTailBasicBlock()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfo_.aicNum ? 0UL : mnCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;

    if (tailCnt != 0UL) {
        while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= compileInfo_.aicNum) {
            runInfo_.tailInfo.mCnt += 1UL;
            if (runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
                runInfo_.tailInfo.nCnt += 1UL;
            }
        }
    }
}

uint64_t MatMulV3AswTiling::GetOuterAxisTailCnt(
    const uint64_t x, const uint64_t y, const uint64_t baseX, const uint64_t baseY, const uint64_t aicNum) const
{
    uint64_t xCnt = MathUtil::CeilDivision(x, baseX);
    uint64_t yCnt = MathUtil::CeilDivision(y, baseY);
    uint64_t xTail = x % baseX;

    uint64_t totalWindows = MathUtil::CeilDivision(xCnt * yCnt, aicNum);
    uint64_t mainWindows = MathUtil::CeilDivision((xCnt - 1UL) * yCnt + yCnt % aicNum, aicNum);
    uint64_t tailWindows = totalWindows - mainWindows;
    uint64_t perfRes = mainWindows * baseX + tailWindows * xTail;

    uint64_t baseTailCntMax = 1UL;
    if (yCnt % aicNum != 0UL) {
        baseTailCntMax = std::min((baseX - xTail) / BASIC_BLOCK_SIZE_16, xCnt);
    }

    uint64_t baseTailCnt = 1UL;
    for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
        uint64_t newTailMain =
            MathUtil::Align(MathUtil::CeilDivision((mergeLen * baseX + xTail), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
        uint64_t newTailLast = mergeLen * (baseX - newTailMain) + xTail;
        uint64_t newMainRound = 0UL;
        uint64_t newTailRound = 0UL;
        if (mergeLen < xCnt - 1UL) {
            // 按照最差的场景计算合并后的主轮，所以性能不会最优
            newMainRound =
                MathUtil::CeilDivision((xCnt - 1UL - mergeLen) * yCnt + (mergeLen + 1UL) * yCnt % aicNum, aicNum);
        }
        if (mergeLen > 0UL) {
            newTailRound =
                std::min(MathUtil::CeilDivision(mergeLen * yCnt + yCnt % aicNum, aicNum), totalWindows - newMainRound);
        }
        uint64_t curPerf = newMainRound * baseX + newTailRound * newTailMain +
                           (totalWindows - newMainRound - newTailRound) * newTailLast;
        if (curPerf < perfRes) {
            perfRes = curPerf;
            baseTailCnt = mergeLen + 1UL;
        }
    }
    return baseTailCnt;
}

void MatMulV3AswTiling::OptimizeEdgeBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore * nCore < compileInfo_.aicNum || mCore == 1UL || nCore == 1UL) {
        return;
    }
    uint64_t mBaseTail = args_.mValue % runInfo_.baseM;
    uint64_t nBaseTail = args_.nValue % runInfo_.baseN;

    bool balanceAfterFixp = args_.kValue <= BASIC_BLOCK_SIZE_256 && args_.nValue % BLOCK_BYTE_SIZE == 0UL;
    if (mBaseTail > 0UL && !args_.isATrans && (nBaseTail == 0UL || mBaseTail <= nBaseTail || balanceAfterFixp)) {
        runInfo_.tailInfo.mBaseTailCnt =
            GetOuterAxisTailCnt(args_.mValue, args_.nValue, runInfo_.baseM, runInfo_.baseN, compileInfo_.aicNum);
    } else if (nBaseTail > 0UL && args_.isBTrans && !balanceAfterFixp) {
        runInfo_.tailInfo.nBaseTailCnt =
            GetOuterAxisTailCnt(args_.nValue, args_.mValue, runInfo_.baseN, runInfo_.baseM, compileInfo_.aicNum);
    }
}

void MatMulV3AswTiling::FormulateBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore * nCore >= compileInfo_.aicNum) {
        runInfo_.baseM = std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
        runInfo_.baseN = std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN);
        return;
    }
    if (mCore <= nCore) {
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
    } else {
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
    }

    while (runInfo_.baseN >= runInfo_.baseM * NUM_TWO && nCore < runInfo_.usedCoreNum / NUM_TWO) {
        nCore = nCore * NUM_TWO;
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }

    while (runInfo_.baseM >= runInfo_.baseN * NUM_TWO && mCore < runInfo_.usedCoreNum / NUM_TWO) {
        mCore = mCore * NUM_TWO;
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }
    mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.usedCoreNum = mCore * nCore;
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), BASIC_BLOCK_SIZE_16);
    runInfo_.baseK = std::min(kValueAlign, kValueMax);
}

ge::graphStatus MatMulV3AswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    OptimizeEdgeBasicBlock();
    FormulateBasicBlock();
    CalcTailBasicBlock();
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    if (MatMulV3TilingHelper::CheckIfDoubleAswt(compileInfo_, args_, 1UL)) {
        aswtModel_ = MatMulV3Model::DOUBLE_ASWT;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3AswTiling::GetTilingKey() const
{
    return MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(aswtModel_)
        .SetL0C2Out(MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_))
        .GetTilingKey();
}
} // namespace matmul_v3_advanced
} // namespace optiling
