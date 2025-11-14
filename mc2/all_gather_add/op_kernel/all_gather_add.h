/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_add.h
 * \brief
 */
#ifndef ALL_GATHER_ADD_H
#define ALL_GATHER_ADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_add_tiling.h"

constexpr int32_t ALLGATHER_ADD_BUFFER_NUM = 1;

namespace AscendC {
class AllGatherAdd {
public:
    __aicore__ inline AllGatherAdd(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM, 
                                GM_ADDR workspaceGM, GM_ADDR contextGM, AllGatherAddTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcclPrepare();
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void HcclFinalize();

private:

    AllGatherAddTilingData *tilingData_;

    TPipe *tPipe_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    TQue<QuePosition::VECIN, ALLGATHER_ADD_BUFFER_NUM> inputQueueGather;
    TQue<QuePosition::VECIN, ALLGATHER_ADD_BUFFER_NUM> inputQueueB;
    TQue<QuePosition::VECOUT, ALLGATHER_ADD_BUFFER_NUM> outputQueueC;

    GlobalTensor<half> inputAGM;
    GlobalTensor<half> gatherOutGM;
    GlobalTensor<half> inputBGM;
    GlobalTensor<half> outputCGM;

    int64_t blockLength_ = 0;
    int64_t tileNum_ = 0;
    uint32_t tileLength_ = 0;

    HcclHandle handleId_{ INVALID_HANDLE_ID };
};

__aicore__ inline void AllGatherAdd::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM, 
                                          GM_ADDR workspaceGM, GM_ADDR contextGM, AllGatherAddTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    blockLength_ = tilingData->blockLength;
    tileLength_ = tilingData->tileLength;
    tileNum_ = tilingData->tileNum;

    // 初始化hccl对象
    // hccl_.InitV2(contextGM, tilingData);
    // hccl_.SetCcTilingV2(offsetof(AllGatherAddTilingData, mc2CcTiling)); // 相对于Mc2InitTiling起始地址的偏移

    // inputAGM.SetGlobalBuffer((__gm__ half*)aGM, tilingData->gatherTileLength);
    gatherOutGM.SetGlobalBuffer((__gm__ half*)gatherGM + blockLength_ * AscendC::GetBlockIdx() * sizeof(half), blockLength_);
    inputBGM.SetGlobalBuffer((__gm__ half*)bGM + blockLength_ * AscendC::GetBlockIdx() * sizeof(half), blockLength_);
    outputCGM.SetGlobalBuffer((__gm__ half*)cGM + blockLength_ * AscendC::GetBlockIdx() * sizeof(half), blockLength_);
    
    tPipe_->InitBuffer(inputQueueGather, ALLGATHER_ADD_BUFFER_NUM, tileLength_ * sizeof(half));
    tPipe_->InitBuffer(inputQueueB, ALLGATHER_ADD_BUFFER_NUM, tileLength_ * sizeof(half));
    tPipe_->InitBuffer(outputQueueC, ALLGATHER_ADD_BUFFER_NUM, tileLength_ * sizeof(half));
}

__aicore__ inline void AllGatherAdd::HcclPrepare()
{
    // 下发通信任务
    // sendBuf recvBuf(数据个数等于sendCount*rank size) sendCount（参与allgather的sendbuf的数据个数） dataType strideCount repeat
    handleId_ = hccl_.AllGather<true>((__gm__ uint8_t*)this->inputAGM.GetPhyAddr(), (__gm__ uint8_t*)this->gatherOutGM.GetPhyAddr(), tilingData_->gatherTileLength,
                                      HcclDataType::HCCL_DATA_TYPE_FP16, 0, tilingData_->commTurn);
}

__aicore__ inline void AllGatherAdd::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<half> gatherLocal = inputQueueGather.AllocTensor<half>();
    AscendC::LocalTensor<half> bLocal = inputQueueB.AllocTensor<half>();
    AscendC::DataCopy(gatherLocal, gatherOutGM[progress * tileLength_], tileLength_);
    AscendC::DataCopy(bLocal, inputBGM[progress * tileLength_], tileLength_);
    inputQueueGather.EnQue(gatherLocal);
    inputQueueB.EnQue(bLocal);
}

__aicore__ inline void AllGatherAdd::Compute()
{
    AscendC::LocalTensor<half> gatherLocal = inputQueueGather.DeQue<half>();
    AscendC::LocalTensor<half> bLocal = inputQueueB.DeQue<half>();
    AscendC::LocalTensor<half> cLocal = outputQueueC.AllocTensor<half>();
    AscendC::Add(cLocal, gatherLocal, bLocal, tileLength_);
    outputQueueC.EnQue<half>(cLocal);
    inputQueueGather.FreeTensor(gatherLocal);
    inputQueueB.FreeTensor(bLocal);
}

__aicore__ inline void AllGatherAdd::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<half> cLocal = outputQueueC.DeQue<half>();
    AscendC::DataCopy(outputCGM[progress * tileLength_], cLocal, tileLength_);
    outputQueueC.FreeTensor(cLocal);
}

__aicore__ inline void AllGatherAdd::HcclFinalize()
{
    hccl_.Finalize();
}

__aicore__ inline void AllGatherAdd::Process()
{
    // HcclPrepare();
    // for (int i = 0; i < tilingData_->commTurn; i++) {
    //     hccl_.Wait(handleId_);
         for (int j = 0; j < tileNum_; j++) {
                CopyIn(i);
                Compute();
                CopyOut(i);
         }
    // }
    // HcclFinalize();
}
}
#endif
