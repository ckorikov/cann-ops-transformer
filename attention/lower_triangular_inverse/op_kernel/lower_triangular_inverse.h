/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_inverse.h
 * \brief
 */

#ifndef __GATED_DELTA_RULE_INVERSE_KERNEL_H_
#define __GATED_DELTA_RULE_INVERSE_KERNEL_H_

#include "lower_triangular_inverse_utils.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace LowerTriangularInverse {

using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM_V = 2;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;

struct MatrixInfo {
    uint32_t mLen = 32;    // 大矩阵边长
    uint32_t curLen = 16;  // 当前矩阵边长
    uint32_t vecLen = 16;  // base矩阵中的对角线矩阵边长, 单个对角线矩阵仅由AIV处理
};

using aT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, float32_t>;
using MT = matmul::MatmulImpl<aT, bT, cT>;

template <class P>
class LowerTriangularMatrixInversion {
public:
    __aicore__ inline LowerTriangularMatrixInversion(MT &matmul) : mm(matmul) {}
    // __aicore__ inline LowerTriangularMatrixInversion() {}
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void AIVProcess(uint64_t offset);
    __aicore__ inline void InitCube();
    __aicore__ inline void AICProcess(GlobalTensor<P> x, GlobalTensor<P> y, GlobalTensor<P> z, uint64_t l, uint64_t m);
    __aicore__ inline void Process();
    __aicore__ inline void AIVCopyIn(uint64_t offset);
    __aicore__ inline void AIVCompute();
    __aicore__ inline void AIVCopyOut(uint64_t offset);
    __aicore__ inline void UpdateLowerBlock(uint64_t beginAddr, uint64_t leftDown, uint64_t rightDown);
    __aicore__ inline void ClearOut(uint64_t offset);

private:
    MT &mm;
    uint32_t coreNum;
    uint64_t batch;
    GlobalTensor<P> xGm_; // 输入     // 问题： 不能用下划线
    GlobalTensor<P> yGm_; // 输出
    GlobalTensor<P> tmpGm_; // 暂存输出
    TQue<TPosition::VECIN, BUFFER_NUM_V> inQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM_V> outQue_;
    TBuf<TPosition::VECCALC> calcBuf_; // 暂存buffer
    TPipe *pipe_;
    MatrixInfo matrixInfo_;
    const LowerTriangularInverseTilingData *tiling;
};

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::Init(InitParams initParams)
{
    // 1. 绑定GlobalTensor
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.y));
    tmpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));

    // 2. 获取TilingData进行初始化
    tiling = initParams.tilingData;
    matrixInfo_.mLen = tiling->m; // 获取最大矩阵边长
    batch = tiling->batch;
    coreNum = tiling->coreNum;

    // 3. 申请UB
    pipe_ = initParams.tPipeIn;
    pipe_->InitBuffer(inQue_, BUFFER_NUM_V, matrixInfo_.vecLen * matrixInfo_.vecLen * sizeof(P));  // 后续需要修改为动态获取
    pipe_->InitBuffer(outQue_, BUFFER_NUM_V, matrixInfo_.vecLen * matrixInfo_.vecLen * sizeof(P)); // 后续需要修改为动态获取
    pipe_->InitBuffer(calcBuf_, tiling->ubRestBytes); // 预留大小，待后续修改 matrixInfo_.mLen * sizeof(P) * uint32_t(matrixInfo_.vecLen)
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0){
            InitCube();
        }
    }
    SyncAll();
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::Process()
{
    // 输入x [b,n,s,c,c] 前三维相乘，得batch
    

    if ASCEND_IS_AIV {
        uint64_t synID = 0;
        for (uint64_t offset = GetBlockIdx() / 2; offset < batch; offset += coreNum) {
            uint64_t offsetCube = offset * matrixInfo_.mLen * matrixInfo_.mLen;
            // 输出矩阵清零
            ClearOut(offsetCube + GetSubBlockIdx() * matrixInfo_.mLen * matrixInfo_.mLen / 2);
            PipeBarrier<PIPE_MTE3>();

            uint64_t numBase = matrixInfo_.mLen / matrixInfo_.vecLen / 2;
            for (uint64_t blockNum = 0; blockNum < numBase; blockNum++){
                uint64_t beginAddr = offsetCube + blockNum * (matrixInfo_.mLen + 1) * matrixInfo_.vecLen * 2;
                AIVProcess(beginAddr);
                PipeBarrier<PIPE_ALL>();
                if (synID > 0 && synID % 15 == 0){
                    CrossCoreWaitFlag(0x2);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
                synID++;
            }
        }
    }
    // AIV 处理基本下三角求逆
    if ASCEND_IS_AIC {
        uint64_t synID = 0;
        for (uint64_t offset = GetBlockIdx(); offset < batch; offset += coreNum) {
            uint64_t offsetCube = offset * matrixInfo_.mLen * matrixInfo_.mLen;

            for (matrixInfo_.curLen = matrixInfo_.vecLen; matrixInfo_.curLen < matrixInfo_.mLen; matrixInfo_.curLen *= 2){
                uint64_t numBase = matrixInfo_.mLen / matrixInfo_.curLen / 2;

                for (uint64_t blockNum = 0; blockNum < numBase; blockNum++){
                    uint64_t beginAddr = offsetCube + blockNum * (matrixInfo_.mLen + 1) * matrixInfo_.curLen * 2;
                    uint64_t leftDown = beginAddr + matrixInfo_.mLen * matrixInfo_.curLen;
                    uint64_t rightDown = leftDown + matrixInfo_.curLen;
                    if (matrixInfo_.curLen == matrixInfo_.vecLen){
                        if (synID > 0 && synID % 15 == 0){
                            CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
                        }
                        CrossCoreWaitFlag(0x3);
                        synID++;
                    }
                    UpdateLowerBlock(beginAddr, leftDown, rightDown);
                }
            }
        }
    }
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::UpdateLowerBlock(uint64_t beginAddr, uint64_t leftDown, uint64_t rightDown)
{
    // -1 @ 左矩阵左下角 -> 右矩阵左下角
    AICProcess(tmpGm_, xGm_[leftDown], yGm_[leftDown], matrixInfo_.mLen, matrixInfo_.curLen);
    PipeBarrier<PIPE_ALL>();
    // 右矩阵左下角 @ 右矩阵左上角 -> 右矩阵左下角
    AICProcess(yGm_[leftDown], yGm_[beginAddr], yGm_[leftDown], matrixInfo_.mLen, matrixInfo_.curLen);
    PipeBarrier<PIPE_ALL>();
    // 右矩阵左下角 @ 右矩阵右下角 -> 右矩阵左下角
    AICProcess(yGm_[rightDown], yGm_[leftDown], yGm_[leftDown], matrixInfo_.mLen, matrixInfo_.curLen);
    PipeBarrier<PIPE_ALL>();
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::AIVProcess(uint64_t offset)
{
    // 判断当前处理的是哪块下三角基本块
    int64_t subBlockIdx = GetSubBlockIdx();
    // 计算当前块内对角块偏移量
    int64_t subBlockOffset = subBlockIdx * matrixInfo_.vecLen * matrixInfo_.mLen + subBlockIdx * matrixInfo_.vecLen;
    uint64_t offsetAll = offset + subBlockOffset;
    AIVCopyIn(offsetAll);
    AIVCompute();
    AIVCopyOut(offsetAll);
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::AICProcess(GlobalTensor<P> x, GlobalTensor<P> y,
                                                                     GlobalTensor<P> z, uint64_t l, uint64_t m)
{
    mm.SetOrgShape(l, l, l);    // MNK
    mm.SetSingleShape(m, m, m); // SingleCOreMNK
    mm.SetTensorA(x);
    mm.SetTensorB(y);
    mm.IterateAll(z);
    mm.End();
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::AIVCopyIn(uint64_t offset)
{
    // 处理对角线上的下三角矩阵
    LocalTensor<P> xLocal = inQue_.AllocTensor<P>();
    // 拷入当前基本块大小的矩阵
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(matrixInfo_.vecLen); // 行数
    copyParams.blockLen = static_cast<uint32_t>(matrixInfo_.vecLen * sizeof(P));
    copyParams.srcStride = static_cast<uint32_t>((matrixInfo_.mLen - matrixInfo_.vecLen) * sizeof(P)); // 相邻块的间隔
    copyParams.dstStride = static_cast<uint32_t>(0);                                                   // 相邻块的间隔
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

    DataCopyPad(xLocal, xGm_[offset], copyParams, copyPadParams);

    inQue_.EnQue(xLocal);
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::AIVCompute()
{
    LocalTensor<P> yLocal = outQue_.AllocTensor<P>();
    LocalTensor<P> xLocal = inQue_.DeQue<P>();

    auto buff = calcBuf_.Get<P>();
    PipeBarrier<PIPE_V>();
    // 下三角计算
    for (int i = 0; i < matrixInfo_.vecLen; ++i) {
        // 每行的总和sum
        auto sum = buff[0];
        Duplicate(sum, static_cast<P>(0.0), matrixInfo_.vecLen); // 存储的数据清零, 是否有更好的办法？
        auto row = buff[matrixInfo_.vecLen];                     // 再取buff后面一块做暂存
        // 求从k=0到k=i-1的l_ik * xk的总和 -> 对应公式中的sum
        for (int k = 0; k < i; ++k) {
            // 取 L[i][k] 并广播到一行
            Duplicate(row, xLocal.GetValue(i * matrixInfo_.vecLen + k), matrixInfo_.vecLen);
            // 取当前 xi 和 L[i][k]相乘
            MulAddDst(sum, row, yLocal[k * matrixInfo_.vecLen], matrixInfo_.vecLen);
            // 重复 i-1 次达到求和效果
        }
        // 单位矩阵对应行的向量
        auto ei = buff[2 * matrixInfo_.vecLen];
        Duplicate(ei, static_cast<P>(0.0), matrixInfo_.vecLen);
        ei.SetValue(i, static_cast<P>(1.0));

        // 取xi
        auto xi = yLocal[i * matrixInfo_.vecLen];
        // I - SUM = xi
        Sub(xi, ei, sum, matrixInfo_.vecLen);

        // Lik * xk = I - SUM -> xk = (I - SUM) / Lik
        // 此时 k = i
        Duplicate(row, xLocal.GetValue(i * matrixInfo_.vecLen + i), matrixInfo_.vecLen);
        Div(xi, xi, row, matrixInfo_.vecLen);
    }

    PipeBarrier<PIPE_V>();
    inQue_.FreeTensor(xLocal);
    // 写回队列
    outQue_.EnQue(yLocal);
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::AIVCopyOut(uint64_t offset)
{
    LocalTensor<P> yLocal = outQue_.DeQue<P>();

    // 拷出当前基本块大小的矩阵 复用
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(matrixInfo_.vecLen);                                 // 行数
    copyParams.blockLen = static_cast<uint32_t>(matrixInfo_.vecLen * sizeof(P));                       // 每个连续数据块长度，单位长度 32B
    copyParams.srcStride = static_cast<uint32_t>(0);                                                   // 相邻块的间隔
    copyParams.dstStride = static_cast<uint32_t>((matrixInfo_.mLen - matrixInfo_.vecLen) * sizeof(P)); // 相邻块的间隔
    DataCopyPad(yGm_[offset], yLocal, copyParams);

    PipeBarrier<PIPE_V>();
    outQue_.FreeTensor(yLocal);
    PipeBarrier<PIPE_V>();
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::InitCube()
{
    // 以当前地址为起点, 初始化对应m大小, 数值为-1的单位矩阵, 目的地址固定为yGM_
    auto buff = calcBuf_.Get<P>();
    auto ei = buff[4 * matrixInfo_.vecLen]; // 复用地址, 注意偏移
    Duplicate(ei, static_cast<P>(0.0), matrixInfo_.mLen);
    ei.SetValue(0, static_cast<P>(-1.0));
    DataCopy(tmpGm_, ei, matrixInfo_.mLen);
    for (int i = 1; i < matrixInfo_.mLen; ++i)
    {
        ei.SetValue(i - 1, static_cast<P>(0.0));
        ei.SetValue(i, static_cast<P>(-1.0));
        DataCopy(tmpGm_[i * matrixInfo_.mLen], ei, matrixInfo_.mLen);
    }
}

template <class P>
__aicore__ inline void LowerTriangularMatrixInversion<P>::ClearOut(uint64_t offset)
{
    // 以当前地址为起点, 初始化对应m大小, 数值为-1的单位矩阵, 目的地址固定为yGM_
    auto buff = calcBuf_.Get<P>();
    auto ei = buff[4 * matrixInfo_.vecLen]; // 复用地址, 注意偏移
    Duplicate(ei, static_cast<P>(0.0), matrixInfo_.mLen);
    for (int i = 0; i < matrixInfo_.mLen / 2; ++i)
    {
        DataCopy(yGm_[offset + i * matrixInfo_.mLen], ei, matrixInfo_.mLen);
    }
}

} // namespace LowerTriangularInverse

#endif // __GATED_DELTA_RULE_INVERSE_KERNEL_H_