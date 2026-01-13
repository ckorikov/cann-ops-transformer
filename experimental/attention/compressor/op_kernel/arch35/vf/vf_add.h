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
 * \file vf_add.h
 * \brief
 */

#ifndef VF_ADD_H
#define VF_ADD_H

#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t FLOAT_REP_SIZE = 64;
constexpr uint32_t BTYEALIGNSIZE = 32;
constexpr uint32_t REGSIZE = 256;

struct LoadAlignParam{
    uint32_t dataBlockStride; //非连续对齐搬运内的首与首间的间隔，以32B为单位
    uint32_t repeatStride; //非连续对齐搬运时，地址偏移大小，以32B为单位
    uint32_t offset; //搬运结束后偏移的更新大小，以32B为单位
};
/*regSplitNum —— 单个regtesor能存的行数
  regLeftNum —— 尾块regtensor上的元素数
  loopCnt —— 单个r需要循环的次数
  loopLeft —— 单个r循环后遗留的尾块
  loadAlignParam0、loadAlignParam1
*/
template<typename T>
__simd_vf__ void AddVFImpl(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t regSplitNum,
    uint32_t regLeftNum, uint32_t loopCnt, uint32_t loopLeft, LoadAlignParam loadAlignParam0, LoadAlignParam loadAlignParam1) 
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregAdd;
    MicroAPI::MaskReg mask;
    uint32_t count = FLOAT_REP_SIZE;
    for(uint32_t loop = 0; loop < loopCnt; loop++) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                    (vreg0, src0Addr, loadAlignParam0.dataBlockStride, loadAlignParam0.repeatStride, mask);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                    (vreg1, src0Addr, loadAlignParam1.dataBlockStride, loadAlignParam1.repeatStride, mask);
        MicroAPI::Add(vregAdd, vreg0, vreg1, mask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                    (dstAddr, vregAdd, loadAlignParam0.dataBlockStride, loadAlignParam0.repeatStride, mask);
        loadAlignParam0.repeatStride += loadAlignParam0.offset;
        loadAlignParam1.repeatStride += loadAlignParam1.offset;
    }
    if(loopLeft > 0){
        mask = MicroAPI::UpdateMask<T>(regLeftNum);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                    (vreg0, src0Addr, loadAlignParam0.dataBlockStride, loadAlignParam0.repeatStride, mask);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                    (vreg1, src0Addr, loadAlignParam1.dataBlockStride, loadAlignParam1.repeatStride, mask);
        MicroAPI::Add(vregAdd, vreg0, vreg1, mask);
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_NORMAL>
                    (dstAddr, vregAdd, loadAlignParam0.dataBlockStride, loadAlignParam0.repeatStride, mask);
    }
}

/**
 * @brief AddVF 输入与apt相加
 * @param outputLocal 输出tensor []
 * @param inputLocal 输入tensor [row, col]
 * @param aptLocal apt输入tensor [r]
 * @param srcIdx0 input起始位置
 * @param srcIdx1 apt起始位置
 * @param d d轴总大小
 * @param splitD 核间d轴切分大小
 * @param baseD  核内d轴切分大小
 * @param baseS 核内s轴切分大小
 */
template <typename T>
__aicore__ inline void AddVF(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal, const LocalTensor<T> &aptLocal,
    uint32_t srcIdx0, uint32_t srcIdx1, uint32_t d, uint32_t splitD, uint32_t baseD, uint32_t baseS) 
{
    uint32_t regSplitNum= FLOAT_REP_SIZE / baseD;
    uint32_t loopCnt = baseS / regSplitNum; 
    uint32_t loopLeft = baseS - loopCnt * regSplitNum;
    uint32_t regLeftNum = loopLeft * baseD;

    LoadAlignParam loadAlignParam0;
    LoadAlignParam loadAlignParam1;
    loadAlignParam0.dataBlockStride = (splitD / 2) * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam0.repeatStride = srcIdx0 * sizeof(T)/BTYEALIGNSIZE;
    loadAlignParam0.offset = regSplitNum * splitD * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam1.dataBlockStride = (d - baseD) / 2 * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam1.repeatStride = srcIdx1 * sizeof(T) / BTYEALIGNSIZE;
    loadAlignParam1.offset = regSplitNum * d * sizeof(T) / BTYEALIGNSIZE;

    __ubuf__ T * inputAddr = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ T * aptAddr = (__ubuf__ T *)aptLocal.GetPhyAddr();
    __ubuf__ T * outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr();
    
    AddVFImpl<T>(outputAddr, inputAddr, aptAddr, regSplitNum, regLeftNum, loopCnt, loopLeft, loadAlignParam0, loadAlignParam1);
}

#endif