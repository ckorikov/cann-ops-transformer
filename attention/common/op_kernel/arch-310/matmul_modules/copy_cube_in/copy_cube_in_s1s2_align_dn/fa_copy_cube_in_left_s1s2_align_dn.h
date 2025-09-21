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
 * \file fa_copy_cube_in_left_s1s2_align_dn.h
 * \brief
 */

#ifndef FA_COPY_CUBE_IN_LEFT_S1S2_ALIGN_DN_H
#define FA_COPY_CUBE_IN_LEFT_S1S2_ALIGN_DN_H

#include "../../cube_in_buffer/fa_cube_in_buffer_general.h"
#include "lib/../impl/matmul/resource/cube_in_buffer/cube_in_buffer.h"

namespace AscendC {
namespace Impl {
namespace Detail {
template<typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class FACopyCubeInLeftS1s2AlignDn {
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;
public:
    using InputType = INPUT_TYPE;
    __aicore__ inline FACopyCubeInLeftS1s2AlignDn() = default;
    __aicore__ inline ~FACopyCubeInLeftS1s2AlignDn() = default;

    __aicore__ inline void Init() {
        baseHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseHeight();
        baseWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetBaseWidth();
        orgHeight_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgHeight();
        orgWidth_ = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
    }

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose) {
        srcAddr_ = localMatrix.address_;
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose) {
        MATMUL_MODULE(MatmulTensorInfo)->template SetGlobalTensor<false>(globalMatrix, isTranspose);
        srcGlobalAddr_ = globalMatrix.address_;
    }

    // left matrix only prefetch
    __aicore__ inline LocalTensor<TransT>
    LoadData(int curRow, int curCol, int tileHeight, int tileWidth, int batchNum = -1) {
        LocalTensor<TransT> l1;
        tileWidth_ = tileWidth;
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            FAFlagData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
            if (flag.copyCurrent) {
                l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(flag.leftBufIdx);
                GlobalTensor<TransT> aGlobal;
                aGlobal.SetGlobalBuffer(srcGlobalAddr_);
                Nd2NzParams nd2nzParams;
                nd2nzParams.ndNum = 1;
                nd2nzParams.nValue = tileHeight;
                nd2nzParams.dValue = tileWidth;
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
                nd2nzParams.dstNzC0Stride = Align16Func(tileHeight);
                nd2nzParams.dstNzNStride = 1;
                nd2nzParams.dstNzMatrixStride = 0;
                DataCopy(l1, aGlobal, nd2nzParams);
                MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
                MATMUL_MODULE(CubeInBuffer)->DeQue();
            } else {
                MATMUL_MODULE(CubeInBuffer)->DeQue();
                l1 = MATMUL_MODULE(CubeInBuffer)->GetBuffer(flag.leftBufIdx);
            }
        } else {
            l1.SetAddr(srcAddr_);
        }
        return l1;
    }

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>,
                                         int32_t curRow = 0, int32_t curCol = 0) {}

    __aicore__ inline void Destroy() {
        if constexpr (!PhyPosIsL1(INPUT_TYPE::pos)) {
            MATMUL_MODULE(CubeInBuffer)->Destroy();
            FAFlagData flag = MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData();
            if (flag.copyNext) {
                uint8_t tscmIdx = (1 - flag.leftBufIdx);
                LocalTensor<TransT> l1 = MATMUL_MODULE(CubeInBuffer)->AllocTensor(tscmIdx);
                GlobalTensor<TransT> aGlobal;
                uint64_t actualNextAddr = flag.offsetSign ? (uint64_t)(srcGlobalAddr_ + flag.curNextAddrOffset) :
                                          (uint64_t)(srcGlobalAddr_ - flag.curNextAddrOffset);
                __gm__ SrcT* nextGmAddr_ = reinterpret_cast<__gm__ SrcT*>(actualNextAddr);
                aGlobal.SetGlobalBuffer(nextGmAddr_);
                Nd2NzParams nd2nzParams;
                nd2nzParams.ndNum = 1;
                nd2nzParams.nValue = flag.nextMOrN;
                nd2nzParams.dValue = tileWidth_;
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = MATMUL_MODULE(CopyCubeInParams)->GetOrgWidth();
                nd2nzParams.dstNzC0Stride = Align16Func(flag.nextMOrN);
                nd2nzParams.dstNzNStride = 1;
                nd2nzParams.dstNzMatrixStride = 0;
                DataCopy(l1, aGlobal, nd2nzParams);
                MATMUL_MODULE(CubeInBuffer)->EnQue(l1);
            }
        }
    }

private:
    TBuffAddr srcAddr_;
    __gm__ SrcT* srcGlobalAddr_;
    int32_t baseHeight_;
    int32_t baseWidth_;
    int32_t orgHeight_;
    int32_t orgWidth_;
    int32_t tileWidth_;
    bool isTranspose_;
};
}
}
}

#endif // FA_COPY_CUBE_IN_LEFT_S1S2_ALIGN_DN_H
