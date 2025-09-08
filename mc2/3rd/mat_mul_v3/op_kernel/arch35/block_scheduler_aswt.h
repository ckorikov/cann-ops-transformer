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
 * \file block_scheduler_aswt.h
 * \brief
 */

#ifndef ACT_BLOCK_SCHEDULER_ASWT_BUILTIN_H
#define ACT_BLOCK_SCHEDULER_ASWT_BUILTIN_H

#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/utils/status_utils.h"
#include "mat_mul_tiling_data.h"

namespace Act {
namespace Gemm {
namespace Block {
constexpr static uint16_t B_FULL_LOAD_MODE = 2;
template <
    class ProblemShape_,
    class L1TileShape_,
    class L0TileShape_
>
class BlockSchedulerAswtBuiltIn {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t kTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t batch_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1};
    int64_t tileNum_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t lastTileIdx_{-1};
    int64_t nSplitOffset_{0};
    int64_t mSplitOffset_{0};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};
    int64_t isHf32_{0};
    int64_t l1BuferNum_{0};
    int32_t l0cDB_{1};

    static constexpr uint64_t WINDOW_LEN = 4UL;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const MatMulV3BasicTilingData* tilingData;
    };

    struct BatchMatMulParams {
        const BatchMatMulV3BasicTilingData* tilingData;
    };
public:
    __aicore__ inline BlockSchedulerAswtBuiltIn(const ProblemShape& shape, int64_t blockIdx, int64_t blockNum, const Params& params)
        : blockIdx_(blockIdx), blockNum_(blockNum)
    {
        k_ = shape.k;
        batch_ = AscendC::Std::max(shape.b, 1L);
        mL1_ = params.tilingData->mL1;
        nL1_ = params.tilingData->nL1;
        kL1_ = params.tilingData->kL1;
        baseM_ = params.tilingData->baseM;
        baseN_ = params.tilingData->baseN;
        baseK_ = params.tilingData->baseK;
        isHf32_ = params.tilingData->isHf32;
        l1BuferNum_ = params.tilingData->l1BufferNum;
        l0cDB_ = params.tilingData->l0cDB;
        mTileNum_ = CeilDiv(shape.m, params.tilingData->mL1);
        nTileNum_ = CeilDiv(shape.n, params.tilingData->nL1);
        kTileNum_ = CeilDiv(k_, params.tilingData->kL1);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, batch_);
        tileNum_ = mTileNum_ * nTileNum_;
        int64_t tailTileNum = tileNum_ % blockNum_;
        tailL1M_ = shape.m - (mTileNum_ - 1) * params.tilingData->mL1;
        tailL1N_ = shape.n - (nTileNum_ - 1) * params.tilingData->nL1;

        if (batch_ == 1) {
            mTailCnt_ = params.tilingData->mTailCnt;
            nTailCnt_ = params.tilingData->nTailCnt;
            int64_t mTailSplit = (tailL1M_ + mTailCnt_ - 1) / mTailCnt_;
            int64_t nTailSplit = (tailL1N_ + nTailCnt_ - 1) / nTailCnt_;
            mTailCnt_ = (tailL1M_ + mTailSplit - 1) / mTailSplit;
            nTailCnt_ = (tailL1N_ + nTailSplit - 1) / nTailSplit;
            tailCnt_ = mTailCnt_ * nTailCnt_;
            tileNum_ += (tailCnt_ - 1) * tailTileNum;
        }
        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_ * batch_;
    }

    __aicore__ inline int64_t Gethf32Flag()
    {
        return isHf32_;
    }

    __aicore__ inline int64_t GetL1BuferNum_()
    {
        return l1BuferNum_;
    }

    __aicore__ inline int64_t GetL0cDB()
    {
        return l0cDB_;
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape() {
        return {mL1_, nL1_, kL1_, 1};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape() {
        return {baseM_, baseN_, baseK_, 1};
    }

    __aicore__ inline int64_t GetBlockNum(ProblemShape shape, int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (tileNum_ * batch_ < blockNum) {
            tilingBlockNum = tileNum_ * batch_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t blkM = (mTileIdx_ == (mTileNum_ - 1)) ? tailL1M_ : mL1_;
        int64_t blkN = (nTileIdx_ == (nTileNum_ - 1)) ? tailL1N_ : nL1_;
        if (tileIdx / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            return {blkM, blkN, k_, batch_};
        }
        int64_t splitBlkM = CeilDiv(blkM, mTailCnt_);
        int64_t splitBlkN = CeilDiv(blkN, nTailCnt_);
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        mSplitOffset_ = mSplitIdx * splitBlkM;
        nSplitOffset_ = nSplitIdx * splitBlkN;
        if (mSplitOffset_ >= blkM || nSplitOffset_ >= blkN) {
            return {0, 0, k_, batch_};
        }
        splitBlkM = AscendC::Std::min(blkM - mSplitOffset_, splitBlkM);
        splitBlkN = AscendC::Std::min(blkN - nSplitOffset_, splitBlkN);
        return {splitBlkM, splitBlkN, k_, batch_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t batchIdx = 0;
        if (batch_ > 1) {
            batchIdx = tileIdx / tileNum_;
        }

        return {mTileIdx_ * mL1_ + mSplitOffset_,
                nTileIdx_ * nL1_ + nSplitOffset_,
                0,
                batchIdx};
    }

private:
    __aicore__ inline void UpdateMNTileIdx(int64_t tmpIdx)
    {
        if (lastTileIdx_ == tmpIdx) {
            return;
        }
        lastTileIdx_ = tmpIdx;

        int64_t tileIdx = tmpIdx % tileNum_;
        if (tileIdx / blockNum_ == (perCoreBlockNum_ - 1) && tailCnt_ > 1) {
            tileIdx = (perCoreBlockNum_ - 1) * blockNum_ + blockIdx_ / tailCnt_;
        }
        int64_t rowIdx = tileIdx / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx % mainWindow_;
            nTileIdx_ = (tileIdx / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = tileIdx - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx % 2 != 0) { // 2: mode 2 means even row, need reverse scan
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }
};

template <
    class ProblemShape_,
    class L1TileShape_,
    class L0TileShape_,
    bool TransA_,
    bool TransB_>
struct BlockSchedulerSelector<
    ProblemShape_,
    L1TileShape_,
    L0TileShape_,
    Act::Gemm::BuiltInAswtScheduler<>,
    TransA_,
    TransB_
> {
  using SchedulerOp = BlockSchedulerAswtBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

template <
    class ProblemShape_,
    class L1TileShape_,
    class L0TileShape_,
    bool TransA_,
    bool TransB_>
struct BlockSchedulerSelector<
    ProblemShape_,
    L1TileShape_,
    L0TileShape_,
    Act::Gemm::BuiltInAswtScheduler<B_FULL_LOAD_MODE>,
    TransA_,
    TransB_
> {
  using SchedulerOp = BlockSchedulerAswtBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Act
#endif