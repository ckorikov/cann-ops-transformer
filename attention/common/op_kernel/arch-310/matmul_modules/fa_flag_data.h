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
 * \file fa_flag_data.h
 * \brief
 */

#ifndef FA_FLAG_DATA_H_
#define FA_FLAG_DATA_H_

#include "lib/matmul_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
struct FAFlagData {
    uint64_t curNextAddrOffset : 40;  // 当前与下次搬运的地址偏移
    uint64_t offsetSign : 1;  // 下次搬运地址是否比当前地址大. 比当前大的为1，否则是0
    uint64_t reuseLeft : 1;  // 是否复用左矩阵
    uint64_t reuseRight : 1;  // 是否复用右矩阵
    uint64_t copyCurrent : 1;  // 在不复用的前提下，是否搬运当前轮次的MTE2
    uint64_t copyNext : 1;  // 在不复用的前提下，是否搬运下一次的MTE2
    uint64_t leftBufIdx : 2;  // 左矩阵的index
    uint64_t rightBufIdx : 3;  // 右矩阵的index
    uint64_t nextMOrN : 9;  // 下一次预取的M或者N
    // m方向是否有additional数据，sameAB模式下TailM只能为偶数，所以当实际M为奇数的时候(TailM = 实际M + 1)，我们要通过这个字段来还原实际M的值
    uint64_t mOrNAdditionalSize : 5;
};

struct FaPaFlagData {
    uint32_t bIdx : 17; // max b = 65536, 17bit
    uint32_t nIdx : 9; // max n = 256, 9bit
    uint32_t resv0 : 6;
    uint32_t blockTableDim2;
    uint32_t blockSize : 10; // max blockSize = 512, 10bit
    uint32_t kHeadSize : 10; // max kHeadSize = 512, 10bit
    uint32_t vHeadSize : 10; // max vHeadSize = 512, 10bit
    uint32_t resv1 : 2;
    uint32_t kvHeadNum : 9; // max kvHeadNum = 256, 9bit
    uint32_t isLayoutBSH : 1; // kv cache layout bbh or bnbd
    uint32_t splitD : 1;
    uint32_t reuseLeft : 1;     // 是否复用左矩阵
    uint32_t reuseRight : 1;  // 是否复用右矩阵, 暂不生效
    uint32_t leftBufIdx : 3;    // 左矩阵的buf index
    uint32_t rightBufIdx : 3;   // 右矩阵的buf index
    // m方向是否有additional数据，sameAB模式下TailM只能为偶数
    // 所以当实际M为奇数的时候（TailM=实际M+1），我们要通过这个字段来还原实际的M的值
    uint32_t mOrNAdditionalSize : 1;
    uint32_t nextMOrN : 9;  // 下一次预取的M或者N, 暂不生效
    uint32_t resv2 : 3;
    int64_t s2SingleOffset;
    uint64_t tensorBAddr;
    uint64_t blockTableAddr;
    uint32_t paBlockNumSum;
};

template<bool hasRope>
struct FAFlagDataWhole {
    FAFlagData baseFlag;
};

template<>
struct FAFlagDataWhole<false> {
    FAFlagData baseFlag;
};

template<>
struct FAFlagDataWhole<true> {
    FAFlagData baseFlag;
    uint64_t qRopeAddr;  // query_rope address
    uint64_t kRopeAddr;  // key_rope address
    uint64_t ropeOffset;  // address offset
    uint32_t ropeKa;  // mm1 query_rope stride
    uint32_t ropeKb;  // mm1 key_rope stride
    uint16_t dSize;  // query dSize
    uint16_t dSizeRope;  // query_rope dSize
};

static constexpr AscendC::TQueConfig TSCM_CONFIG = {.nd2nz = false,
    .nz2nd = false,
    .scmBlockGroup = true,
    .bufferLen = 0,
    .bufferNumber = 1,
    .consumerSize = 0,
    .consumer = {},
    .enableStaticEvtId = true};

// L1 global variable
struct GlobalTscmArray {
    __aicore__ inline GlobalTscmArray () {};
#ifndef __CCE_KT_TEST__
    AscendC::TSCM<AscendC::TPosition::GM, 1, &TSCM_CONFIG> localQue[8];
#else
    AscendC::TSCM<TPosition::GM, 1, 0x4> localQue[8];
#endif
};
__BLOCK_LOCAL__ __inline__ GlobalTscmArray* tscmGlobal;
}
}
}

#endif // FA_FLAG_DATA_H_
