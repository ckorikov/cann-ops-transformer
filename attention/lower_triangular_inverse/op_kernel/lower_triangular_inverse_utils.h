/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_inverse_utils.h
 * \brief
 */

#ifndef __GATED_DELTA_RULE_INVERSE_KERNEL_UTILS_H_
#define __GATED_DELTA_RULE_INVERSE_KERNEL_UTILS_H_

#include "kernel_operator.h"


namespace LowerTriangularInverse {

using namespace AscendC;

struct InitParams {
  GM_ADDR x;
  GM_ADDR y;
  GM_ADDR workspace;
  TPipe *tPipeIn;
  LowerTriangularInverseTilingData *tilingData;
};

constexpr uint32_t BASE_BLOCK_BYTE = 32; // 分形中的元素个数固定为 A:16*(32B/sizof(T)), B:(32B/sizof(T))*16, C:16*16 
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t MAX_LEN = 128;
constexpr uint32_t BUFFER_NUM = 2;

// 由于高阶API无法支持可变大小矩阵，所以设置自定义矩阵乘法
// 当前算子最大计算量为128 * 128, 全部采用全载的方式
// 注意NPU类型是否支持float，部分NPU无法支持float类型的loadData和mmad函数
template <class P>
class CustMatmul {
public:
    __aicore__ inline void Init(GlobalTensor<P> A, GlobalTensor<P> B, GlobalTensor<P> C, uint64_t l, uint64_t m, TPipe *tPipeIn) {
        A_ = A;
        B_ = B;
        C_ = C;

        // 分形元素个数计算
        CUBE_BLOCK_1 = CUBE_BLOCK;
        CUBE_BLOCK_2 = BASE_BLOCK_BYTE / sizeof(P);
        CUBE_BLOCK_SIZE = CUBE_BLOCK_1 * CUBE_BLOCK_2;   // 单个分形大小
        
        // 原大矩阵的大小
        l_ = l;

        // 当前计算的小矩阵输入仅为方阵
        m_ = m;
        n_ = m_;
        k_ = m_;

        // 分配空间
        pipe_ = tPipeIn;
        pipe_->InitBuffer(inAQue_, BUFFER_NUM, m_ * k_ * sizeof(P));
        pipe_->InitBuffer(inBQue_, BUFFER_NUM, k_ * n_ * sizeof(P));
        pipe_->InitBuffer(inA2Que_, 1, m_ * k_ * sizeof(P));    // 后续待试验, 开启是否会导致内存溢出
        pipe_->InitBuffer(inB2Que_, 1, k_ * n_ * sizeof(P));
        pipe_->InitBuffer(outQue_, 1, m_ * n_ * sizeof(P));
    }
    
    __aicore__ inline void CopyIn() {
        // GM -> L1, ND -> Nz
        LocalTensor<P> a1Local = inAQue_.AllocTensor<P>();
        LocalTensor<P> b1Local = inBQue_.AllocTensor<P>();
        // AB大小相同
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = m_;
        nd2nzParams.dValue = k_;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = l_; // 小矩阵相邻行的偏移, 也就是大矩阵边长
        nd2nzParams.dstNzC0Stride = m_;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy(a1Local, A_, nd2nzParams);
        DataCopy(b1Local, B_, nd2nzParams);
        inAQue_.EnQue<P>(a1Local);
        inBQue_.EnQue<P>(b1Local);
    }

    // 一条边所包含的分形个数计算
    // flag为true时, 默认为分形16的那一条边, 为false时, 为自适应长度的那条边
    __aicore__ inline uint32_t NumCubeBlock(uint32_t len, bool flag = true) {
        return (flag ? (len / CUBE_BLOCK_1) : (len / CUBE_BLOCK_2));
    }

    __aicore__ inline void MoveToL0() {
        LocalTensor<P> a1Local = inAQue_.DeQue<P>();
        LocalTensor<P> a2Local = inA2Que_.AllocTensor<P>();
        LocalTensor<P> b1Local = inBQue_.DeQue<P>();
        LocalTensor<P> b2Local = inB2Que_.AllocTensor<P>();
        
        // A, Nz -> Zz, 按行处理
        // float类型时, 单个分形大小为: 16 * 8
        uint32_t dstOffset = NumCubeBlock(k_, false) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        LoadData2DParams loadAParams;
        loadAParams.repeatTimes = NumCubeBlock(k_, false); // 按列分分形
        loadAParams.srcStride = NumCubeBlock(m_);   // 按行分分形
        loadAParams.dstGap = 0;
        loadAParams.ifTranspose = false;
        for (int i = 0; i < NumCubeBlock(m_); ++i) {
            LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadAParams);
        }
        
        // B, Nz -> Zn, 按行处理
        // float类型时, 单个分形大小为: 8 * 16
        uint32_t dstBOffset = NumCubeBlock(n_) * CUBE_BLOCK_SIZE;
        uint32_t srcBOffset = CUBE_BLOCK_SIZE;
        LoadData2DParams loadBParams;
        loadBParams.repeatTimes = NumCubeBlock(n_); // 按列分分形
        loadBParams.srcStride = NumCubeBlock(k_, false);   // 按行分分形
        loadBParams.dstGap = 0;
        loadBParams.ifTranspose = true; // 分形内部z->n
        for (int i = 0; i < NumCubeBlock(k_, false); ++i) {
            LoadData(b2Local[i * dstBOffset], b1Local[i * srcBOffset], loadBParams);
        }
        
        inA2Que_.EnQue<P>(a2Local);
        inB2Que_.EnQue<P>(b2Local);
        inAQue_.FreeTensor(a1Local);
        inBQue_.FreeTensor(b1Local);
    }

    __aicore__ inline void Compute() {
        LocalTensor<P> a2Local = inA2Que_.DeQue<P>();
        LocalTensor<P> b2Local = inB2Que_.DeQue<P>();
        LocalTensor<P> c1Local = outQue_.AllocTensor<P>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m_;
        mmadParams.n = n_;
        mmadParams.k = k_;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQue_.EnQue<P>(c1Local);
        inA2Que_.FreeTensor(a2Local);
        inB2Que_.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<P> c1Local = outQue_.DeQue<P>();

        // CO1 -> GM, 按列切分
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n_;
        fixpipeParams.mSize = m_;
        fixpipeParams.srcStride = m_; // 源NZ矩阵中相邻Z排布的起始地址偏移, 单位：C0_Size(16*sizeof(T))
        fixpipeParams.dstStride = l_;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0; // 配置为0时不生效
        fixpipeParams.dstNdStride = 0; // 配置为0时不生效

        // 默认使能Nz->ND
        AscendC::Fixpipe(C_, c1Local, fixpipeParams);

        outQue_.FreeTensor(c1Local);
    }

    __aicore__ inline void Process() {
        // CopyA, GM -> L1
        CopyIn();
        MoveToL0();
        Compute();
        CopyOut();

        // // LocalTensor<P> a1Local = inAQue_.DeQue<P>();
        // PipeBarrier<PIPE_V>();
        // DumpTensor(C_, 0, 64);
        // PipeBarrier<PIPE_V>();
        // // inAQue_.FreeTensor(a1Local);
    }

private:
    GlobalTensor<P> A_; // 左矩阵
    GlobalTensor<P> B_; // 右矩阵
    GlobalTensor<P> C_; // 目的矩阵
    TQue<TPosition::A1, 1> inAQue_;
    TQue<TPosition::B1, 1> inBQue_;
    TQue<TPosition::A2, 1> inA2Que_;
    TQue<TPosition::B2, 1> inB2Que_;
    TQue<TPosition::CO1, 1> outQue_;
    uint64_t l_;
    uint64_t m_;
    uint64_t k_;
    uint64_t n_;
    uint64_t CUBE_BLOCK_1; // 分形固定16个元素
    uint64_t CUBE_BLOCK_2;      // 分形非固定元素个数
    uint64_t CUBE_BLOCK_SIZE;   // 分形大小随数据类型变动
    TPipe *pipe_;
};









} // namespace GroupedMatmulFinalizeRouting
#endif // __GATED_DELTA_RULE_INVERSE_KERNEL_UTILS_H_