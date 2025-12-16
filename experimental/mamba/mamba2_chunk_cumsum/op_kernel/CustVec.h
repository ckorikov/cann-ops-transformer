/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "tensorutils.h"
#include "paramutils.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkCumsum {

constexpr int BASEL = 64;
constexpr int BASEN = 64;
constexpr int BASEH = 128;
constexpr int SUB_BASEH = BASEH / 2;
constexpr int BLK_SIZE = BASEL * SUB_BASEH;
constexpr float COMPARE_VALUE = 20.0;
constexpr float CLAMP_MAX = 10000000.0f;

struct CustVecShapeInfo{
    int nstepsH;
    int BCH;
    int BCH_PER_CORE;
    int BCH1;
    int BCH2;
    int C;
    int L;
    int H;
};


__aicore__ inline void tilingShapeCustVec(int B, int C, int H, int L, CustVecShapeInfo &shape){
    shape.nstepsH = (H / BASEH);
    shape.BCH = ((B * C) * shape.nstepsH);
    shape.BCH_PER_CORE = CeilDiv(shape.BCH,GetBlockNum());
    shape.BCH1 = (shape.BCH_PER_CORE * get_block_idx());
    shape.BCH2 = (((shape.BCH_PER_CORE + shape.BCH1))<(shape.BCH)) ? ((shape.BCH_PER_CORE + shape.BCH1)) : (shape.BCH);
    shape.C = C;
    shape.L = L;
    shape.H = H;
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR at_mtx_, GM_ADDR dt_mtx_, GM_ADDR dtbias_mtx_, GM_ADDR dtmask_mtx_, GM_ADDR out_mtx_, GM_ADDR out1_mtx_, GM_ADDR out2_mtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        at_mtx.SetGlobalBuffer((__gm__ float*) at_mtx_);
        dt_mtx.SetGlobalBuffer((__gm__ half*) dt_mtx_);
        dtbias_mtx.SetGlobalBuffer((__gm__ half*) dtbias_mtx_);
        dtmask_mtx.SetGlobalBuffer((__gm__ half*) dtmask_mtx_);
        out_mtx.SetGlobalBuffer((__gm__ float*) out_mtx_);
        out1_mtx.SetGlobalBuffer((__gm__ float*) out1_mtx_);
        out2_mtx.SetGlobalBuffer((__gm__ float*) out2_mtx_);
        
        // Local buffers
        at_buf.Init(SUB_BASEH);
        dt_buf.Init(BLK_SIZE);
        dtbias_buf.Init(SUB_BASEH);
        dtmask_buf.Init(BLK_SIZE);
        out1buf.Init(BLK_SIZE);
        out2buf.Init(BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp0, BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp1, BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp2, SUB_BASEH);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp3, BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(cmp_mask, BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(max_buf, EIGHT);
        cumsum_tensor.Init(SUB_BASEH);
        // Initialize events
        in_ready.Init();
        in_empty.Init();
        out_ready.Init();
        out_empty.Init();
    }
    
    __aicore__ inline void PreCompute(){
        // TODO: User-defined pre-computation
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        
        int cc_cnt = 0;
        auto castParamsH2F = CastHalf2FloatRepeatParams();
        auto castParamsF2H = CastFloat2HalfRepeatParams();
        auto unaryParams = MakeDefaultUnaryRepeatParams();
        auto binaryParams = MakeDefaultBinaryRepeatParams();

        for (int bch=shape.BCH1; bch<shape.BCH2; ++bch){
            int b = (bch / (shape.C * shape.nstepsH));
            int c = ((bch % (shape.C * shape.nstepsH)) / shape.nstepsH);
            int h = (((bch % (shape.C * shape.nstepsH)) % shape.nstepsH) * BASEH);
            for (int l=0; l<shape.L; l+=BASEL){
                
                in_empty.wait();
                GM2UB(at_buf.get(cc_cnt), at_mtx[(h + (get_subblockid() * SUB_BASEH))], 1, (int)SUB_BASEH/MTE_FLOAT, 0, 0);
                GM2UB(dt_buf.get(cc_cnt), dt_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + ((get_subblockid() * SUB_BASEH))))], BASEL, (int)SUB_BASEH/MTE_HALF, ((shape.H - SUB_BASEH) / MTE_HALF), 0);
                GM2UB(dtbias_buf.get(cc_cnt), dtbias_mtx[(h + ((get_subblockid() * SUB_BASEH)))], 1, SUB_BASEH/MTE_HALF, 0, 0);
                GM2UB(dtmask_buf.get(cc_cnt), dtmask_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + ((get_subblockid() * SUB_BASEH))))], BASEL, SUB_BASEH/MTE_HALF, ((shape.H - SUB_BASEH) / MTE_HALF), 0);
                in_ready.set();
                
                out_empty.wait();
                in_ready.wait();
                if ((cc_cnt == 0)){
                    Duplicate<float, false>(max_buf, 0.000000f, MASK_PLACEHOLDER, 1, 1, N_DBLK_FLOAT);
                    PipeBarrier<PIPE_V>();
                    Adds<float, false>(max_buf, max_buf, CLAMP_MAX, MASK_PLACEHOLDER, 1, {0, 0, 0, 0});
                    PipeBarrier<PIPE_V>();
                }
                Cast<float, half, false>(cc_tmp1, dt_buf.get(cc_cnt), RoundMode::CAST_NONE, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, castParamsH2F);
                Cast<float, half, false>(cc_tmp2, dtbias_buf.get(cc_cnt), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, castParamsH2F);
                Cast<float, half, false>(cc_tmp3, dtmask_buf.get(cc_cnt), RoundMode::CAST_NONE, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, castParamsH2F);
                PipeBarrier<PIPE_V>();
                auto custparam = MakeDefaultBinaryRepeatParams();
                custparam.src1RepStride = 0; // {1,1,1,8,8,0}
                Add<float, false>(cc_tmp1, cc_tmp1, cc_tmp2, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, custparam);
                PipeBarrier<PIPE_V>();
                CompareScalar<float, uint8_t>(cmp_mask, cc_tmp1, COMPARE_VALUE, CMPMODE::LT, BLK_SIZE);
                PipeBarrier<PIPE_V>();
                UB2UB(cc_tmp0, cc_tmp1, BASEL, SUB_BASEH/MTE_FLOAT, 0, 0);
                PipeBarrier<PIPE_V>();
                Exp<float, false>(cc_tmp1, cc_tmp1, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, unaryParams);
                PipeBarrier<PIPE_V>();
                Adds<float, false>(cc_tmp1, cc_tmp1, 1.0f, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, unaryParams);
                PipeBarrier<PIPE_V>();
                Ln<float, false>(cc_tmp1, cc_tmp1, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, unaryParams);
                PipeBarrier<PIPE_V>();
                Select<float, uint8_t, false>(cc_tmp1, cmp_mask, cc_tmp1, cc_tmp0, SELMODE::VSEL_TENSOR_TENSOR_MODE, VEC_FLOAT, BLK_SIZE/VEC_FLOAT, binaryParams);
                PipeBarrier<PIPE_V>();
                CompareScalar<float, uint8_t>(cmp_mask, cc_tmp1, CLAMP_MAX, CMPMODE::LT, BLK_SIZE);
                PipeBarrier<PIPE_V>();
                Select<float, uint8_t>(cc_tmp1, cmp_mask, cc_tmp1, static_cast<float>(CLAMP_MAX.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, BLK_SIZE);
                PipeBarrier<PIPE_V>();
                Mul<float, false>(out1buf.get(cc_cnt), cc_tmp1, cc_tmp3, MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, binaryParams);
                PipeBarrier<PIPE_V>();
                Mul<float, false>(cc_tmp0, out1buf.get(cc_cnt), at_buf.get(cc_cnt), MASK_PLACEHOLDER, BLK_SIZE/VEC_FLOAT, custparam);
                PipeBarrier<PIPE_V>();
                if ((l > 0)){
                    Add<float, false>(cc_tmp0, cc_tmp0, cumsum_tensor.get(cc_cnt), MASK_PLACEHOLDER, SUB_BASEH/VEC_FLOAT, {1, 1, 1, 0, 0, 0});
                    PipeBarrier<PIPE_V>();
                }
                for (int l1=0; l1<BASEL-1; ++l1){
                    Add<float, false>(cc_tmp0[((l1 + 1) * SUB_BASEH)], cc_tmp0[(l1 * SUB_BASEH)], cc_tmp0[((l1 + 1) * SUB_BASEH)], MASK_PLACEHOLDER, 1, {1, 1, 1, 0, 0, 0});
                    PipeBarrier<PIPE_V>();
                }
                if ((l < shape.L)){
                    UB2UB(cumsum_tensor.get((cc_cnt + 1)), cc_tmp0[(BASEL-1)*SUB_BASEH], 1, SUB_BASEH/MTE_FLOAT, 0, 0);
                    PipeBarrier<PIPE_V>();
                }
                UB2UB(out2buf.get(cc_cnt), cc_tmp0, BASEL, SUB_BASEH/MTE_FLOAT, 0, 0);
                PipeBarrier<PIPE_V>();
                in_empty.set();
                
                out_ready.set();
                out_ready.wait();
                UB2GM(out_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + (get_subblockid() * SUB_BASEH)))], out1buf.get(cc_cnt), BASEL, SUB_BASEH/MTE_FLOAT, 0, ((shape.H - SUB_BASEH) / MTE_FLOAT));
                UB2GM(out1_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + (get_subblockid() * SUB_BASEH)))], out2buf.get(cc_cnt), BASEL, SUB_BASEH/MTE_FLOAT, 0, ((shape.H - SUB_BASEH) / MTE_FLOAT));
                if (((l + 64) >= shape.L)){
                    UB2GM(out2_mtx[((((b * shape.C) + c) * shape.H) + (h + (get_subblockid() * SUB_BASEH)))], out2buf.get(cc_cnt)[(BASEL-1)*SUB_BASEH], 1, SUB_BASEH/MTE_FLOAT, 0, ((shape.H - SUB_BASEH) / MTE_FLOAT));
                }
                out_empty.set();
                
                cc_cnt = (cc_cnt + 1);
            }
        }
        
        in_empty.release();
        out_empty.release();
    }
    
private:
    CustVecShapeInfo shape;
    // Global Tensors
    GlobalTensor<float> at_mtx;
    GlobalTensor<half> dt_mtx;
    GlobalTensor<half> dtbias_mtx;
    GlobalTensor<half> dtmask_mtx;
    GlobalTensor<float> out_mtx;
    GlobalTensor<float> out1_mtx;
    GlobalTensor<float> out2_mtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> at_buf;
    DBuff<half, TPosition::VECCALC> dt_buf;
    DBuff<half, TPosition::VECCALC> dtbias_buf;
    DBuff<half, TPosition::VECCALC> dtmask_buf;
    DBuff<float, TPosition::VECCALC> out1buf;
    DBuff<float, TPosition::VECCALC> out2buf;
    LocalTensor<float> cc_tmp0;
    LocalTensor<float> cc_tmp1;
    LocalTensor<float> cc_tmp2;
    LocalTensor<float> cc_tmp3;
    LocalTensor<uint8_t> cmp_mask;
    LocalTensor<float> max_buf;
    DBuff<float, TPosition::VECCALC> cumsum_tensor;
    
    // Double events
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    // User-defined events
};
// Auto-generated code. Readability is not guaranteed
} // namespace Mambav2ChunkCumsum
} // namespace npu_ops_transformer_ext