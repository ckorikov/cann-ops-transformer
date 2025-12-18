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

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkState {

struct CustCubeShapeInfo{
    int BCH;
    int BCH_PER_CORE;
    int BCH1;
    int BCH2;
    int C;
    int L;
    int H;
    int G;
    int N;
    int P;
    int BASEK;
};


__aicore__ inline void tilingShapeCustCube(int B, int C, int H, int G, int L, int N, int P, CustCubeShapeInfo &shape){
    shape.BCH = ((int)((B * C) * H) / BASEH);
    shape.BCH_PER_CORE = CeilDiv(shape.BCH,GetBlockNum());
    shape.BCH1 = (shape.BCH_PER_CORE * get_block_idx());
    shape.BCH2 = (((shape.BCH_PER_CORE + shape.BCH1))<(shape.BCH)) ? ((shape.BCH_PER_CORE + shape.BCH1)) : (shape.BCH);
    shape.C = C;
    shape.L = L;
    shape.H = H;
    shape.G = G;
    shape.N = N;
    shape.P = P;
    shape.BASEK = ((shape.L)<(CBASEK)) ? (shape.L) : (CBASEK);
}


class CustCube{
public:
    __aicore__ inline CustCube(){}
    __aicore__ inline void Init(GM_ADDR vec_out_, GM_ADDR xt_mtx_, GM_ADDR out_mtx_, CustCubeShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        OccupyMMTE1Events();
        // Global Tensors
        vec_out.SetGlobalBuffer((__gm__ half*) vec_out_);
        xt_mtx.SetGlobalBuffer((__gm__ half*) xt_mtx_);
        out_mtx.SetGlobalBuffer((__gm__ float*) out_mtx_);
        // L1 Buffers
        cs_l1a.Init((shape.BASEK * CBASEM));
        cs_l1b.Init((shape.BASEK * CBASEN));
        // L0A Buffers
        cs_l0a.Init((shape.BASEK * CBASEM));
        // L0B Buffers
        cs_l0b.Init((shape.BASEK * CBASEN));
        // L0C Buffers
        cs_l0c.Init(CBASEM * CBASEN);
        // UB Buffers
        // Initialize events
        l1_empty.Init();
        l1_ready.Init();
        l0_empty.Init();
        l0_ready.Init();
        out_empty.Init();
        out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        l1_empty.setall();
        l0_empty.setall();
        out_empty.setall();
        int cs_input_cnt = 0;
        int cs_output_cnt = 0;
        int ws_cnt = 0;
        CUBE_READY(0, PIPE_FIX);
        CUBE_READY(0, PIPE_FIX);
        CUBE_READY(0, PIPE_FIX);
        for (int bch=shape.BCH1; bch<shape.BCH2; bch+=1){
            int b = ((int)bch / (int)((int)(shape.C * shape.H) / BASEH));
            int c = ((int)(bch % ((int)(shape.C * shape.H) / BASEH)) / (int)((int)shape.H / BASEH));
            int h = ((bch % ((int)(shape.C * shape.H) / BASEH)) % ((int)shape.H / BASEH));
            for (int madn=0; madn<shape.N; madn+=CBASEM){
                WAIT_VEC(0);
                for (int madh=0; madh<8; madh+=1){
                    for (int k=0; k<shape.L; k+=shape.BASEK){
                        l1_empty.wait();
                        L1ND2NZ(cs_l1a.get(cs_input_cnt), vec_out[(((((ws_cnt % 3) * (((GetBlockNum() * 8) * shape.L) * 64)) + (((get_block_idx() * 8) * shape.L) * 64)) + ((madh * shape.L) * 64)) + (k * 64))], shape.BASEK, 64, 64, shape.BASEK);
                        L1ND2NZ(cs_l1b.get(cs_input_cnt), xt_mtx[(((((((b * shape.C) + c) * shape.L) * shape.H) * shape.P) + ((k * shape.H) * shape.P)) + (((h * 8) + madh) * shape.P))], shape.BASEK, 64, (shape.H * shape.P), shape.BASEK);
                        l1_ready.set();
                        l0_empty.wait();
                        l1_ready.wait();
                        L0NZ2NN(cs_l0a.get(cs_input_cnt), cs_l1a.get(cs_input_cnt), shape.BASEK, 64, shape.BASEK, 64);
                        L0NZ2ZN(cs_l0b.get(cs_input_cnt), cs_l1b.get(cs_input_cnt), shape.BASEK, 64, shape.BASEK, 64);
                        l1_empty.set();
                        l0_ready.set();
                        out_empty.wait();
                        l0_ready.wait();
                        MMAD(cs_l0c.get(cs_output_cnt), cs_l0a.get(cs_input_cnt), cs_l0b.get(cs_input_cnt), 64, shape.BASEK, 64, (k == 0), 0);
                        out_ready.set();
                        l0_empty.set();
                        cs_input_cnt = (cs_input_cnt + 1);
                        out_ready.wait();
                        if (((k + shape.BASEK) >= shape.L)){
                            L0C2GM_NZ2ND(out_mtx[(((((((b * shape.C) + c) * shape.H) * shape.N) * shape.P) + ((((h * 8) + madh) * shape.N) * shape.P)) + (madn * shape.P))], cs_l0c.get(cs_output_cnt), 64, 64, shape.P, 64, 0);
                            cs_output_cnt = (cs_output_cnt + 1);
                        }
                        out_empty.set();
                    }
                }
                CUBE_READY(0, PIPE_FIX);
                ws_cnt = (ws_cnt + 1);
            }
        }
        l1_empty.release();
        l0_empty.release();
        out_empty.release();
    }
    
private:
    CustCubeShapeInfo shape;
    // Global Tensors
    GlobalTensor<half> vec_out;
    GlobalTensor<half> xt_mtx;
    GlobalTensor<float> out_mtx;
    // Local buffers
    DBuff<half, TPosition::A1> cs_l1a;
    DBuff<half, TPosition::A1> cs_l1b;
    DBuff<half, TPosition::A2> cs_l0a;
    DBuff<half, TPosition::B2> cs_l0b;
    DBuff<float, TPosition::CO1> cs_l0c;
    
    // Events
    DEvent<PIPE_MTE1, PIPE_MTE2> l1_empty;
    DEvent<PIPE_MTE2, PIPE_MTE1> l1_ready;
    DEvent<PIPE_M, PIPE_MTE1> l0_empty;
    DEvent<PIPE_MTE1, PIPE_M> l0_ready;
    DEvent<PIPE_FIX, PIPE_M> out_empty;
    DEvent<PIPE_M, PIPE_FIX> out_ready;
};

} // Mambav2ChunkState
} // npu_ops_transformer_ext