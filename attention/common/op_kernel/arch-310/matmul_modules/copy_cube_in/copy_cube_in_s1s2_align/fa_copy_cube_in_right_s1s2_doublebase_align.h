/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fa_copy_cube_in_right_s1s2_doublebase_align.h
 * \brief
 */

#ifndef FA_COPY_CUBE_IN_RIGHT_S1S2_DOUBLEBASE_ALIGN_H
#define FA_COPY_CUBE_IN_RIGHT_S1S2_DOUBLEBASE_ALIGN_H

#include "../../cube_in_buffer/fa_cube_in_buffer_double_base.h"
#include "lib/../impl/matmul/resource/cube_in_buffer/cube_in_buffer.h"
#define COPY_NEXT_VALUE false

namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class FACopyCubeInRightS1s2DoubleBaseAlign {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;
public:
    using InputType = INPUT_TYPE;
    __aicore__ inline FACopyCubeInRightS1s2DoubleBaseAlign() = default;
    __aicore__ inline ~FACopyCubeInRightS1s2DoubleBaseAlign() = default;

    __aicore__ inline void Init() {
        baseHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        baseWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        orgHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        isTranspose_ = INPUT_TYPE::isTrans;
        nd2nzParams_.ndNum = 1;
        nd2nzParams_.srcNdMatrixStride = 0;
        nd2nzParams_.dstNzNStride = 1;
        nd2nzParams_.dstNzMatrixStride = 0;
    }

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose) {
        srcAddr_ = localMatrix.address_;
        if constexpr (INPUT_TYPE::isTrans) {
            isTranspose_ = isTranspose;
        }
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose) {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
        if constexpr (INPUT_TYPE::isTrans) {
            isTranspose_ = isTranspose;
        }
    }

    // 这个场景右矩阵是切k的，每次只加载一半
    __aicore__ inline LocalTensor<TransT>
    LoadData(int curRow, int curCol, int tileHeight, int tileWidth, int batchNum = -1) {
        if constexpr (INPUT_TYPE::isTrans) {
            if (isTranspose_) {
                tileHeight = tileHeight ^ tileWidth;
                tileWidth = tileHeight ^ tileWidth;
                tileHeight = tileHeight ^ tileWidth;
                baseHeight_ = MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<true>();
                orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<true>();
            }
        } else {
            baseHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
            orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
        }
        LocalTensor<TransT> l1;
        l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(RIGHT_MATRIX_TSCM_POS * 2 + (callTimes_ & 1));
        GlobalTensor<TransT> aGlobal;
        aGlobal.SetGlobalBuffer(srcGlobalAddr_ + (callTimes_ & 1) * baseHeight_ * orgWidth_);
        nd2nzParams_.nValue = tileHeight;
        nd2nzParams_.dValue = tileWidth;
        nd2nzParams_.srcDValue = orgWidth_;
        nd2nzParams_.dstNzC0Stride = Align16Func(tileHeight);
        DataCopy(l1, aGlobal, nd2nzParams_);
        MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
        MATMUL_MODULE(CubeInBuffer)->DeQue();

        ++callTimes_;
        return l1;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>,
                                         int32_t curRow = 0, int32_t curCol = 0) {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Reset();
        }
    }

    __aicore__ inline void Destroy() {
        callTimes_ = 0;
    }

    __aicore__ inline void Reset() {
        MATMUL_MODULE(CubeInBuffer)->Reset();
    }

private:
    TBuffAddr srcAddr_;
    __gm__ SrcT* srcGlobalAddr_;
    Nd2NzParams nd2nzParams_;
    int64_t callTimes_ = 0;
    int32_t baseHeight_;
    int32_t baseWidth_;
    int32_t orgHeight_;
    int32_t orgWidth_;
    bool isTranspose_{false};
};
}
}
}

#endif // FA_COPY_CUBE_IN_RIGHT_S1S2_DOUBLEBASE_ALIGN_H
