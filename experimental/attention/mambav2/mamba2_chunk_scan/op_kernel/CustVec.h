#pragma once
#include "tensorutils.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkScan {

struct CustVecShapeInfo{
    int BCH;
    int H;
    int G;
    int L;
    int N;
    int P;
    int HPERG;
    int BCH_PERCORE;
    int BCH1;
    int BCH2;
    int BASEL;
};


__aicore__ inline void tilingShapeCustVec(int B, int C, int H, int G, int L, int N, int P, CustVecShapeInfo &shape){
    shape.BCH = ((B * C) * H);
    shape.H = H;
    shape.G = G;
    shape.L = L;
    shape.N = N;
    shape.P = P;
    shape.HPERG = ((int)H / (int)G);
    shape.BCH_PERCORE = CeilDiv(shape.BCH,GetBlockNum());
    shape.BCH1 = (shape.BCH_PERCORE * get_block_idx());
    shape.BCH2 = (((shape.BCH1 + shape.BCH_PERCORE))<(shape.BCH)) ? ((shape.BCH1 + shape.BCH_PERCORE)) : (shape.BCH);
    shape.BASEL = 64;
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR cb_ws_, GM_ADDR dacsmtx_, GM_ADDR dtoutmtx_, GM_ADDR mmtx_, GM_ADDR outmtx_, GM_ADDR statesmtx_, GM_ADDR xmtx_, GM_ADDR dmtx_, GM_ADDR sumoutmtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        cb_ws.SetGlobalBuffer((__gm__ float*) cb_ws_);
        dacsmtx.SetGlobalBuffer((__gm__ float*) dacsmtx_);
        dtoutmtx.SetGlobalBuffer((__gm__ float*) dtoutmtx_);
        mmtx.SetGlobalBuffer((__gm__ half*) mmtx_);
        outmtx.SetGlobalBuffer((__gm__ float*) outmtx_);
        statesmtx.SetGlobalBuffer((__gm__ float*) statesmtx_);
        xmtx.SetGlobalBuffer((__gm__ half*) xmtx_);
        dmtx.SetGlobalBuffer((__gm__ half*) dmtx_);
        sumoutmtx.SetGlobalBuffer((__gm__ float*) sumoutmtx_);
        
        // Local buffers
        dacs_buf1.Init(shape.BASEL);
        AllocateLocalTensor<TPosition::VECCALC>(dacs_brcb, (shape.BASEL * 8));
        AllocateLocalTensor<TPosition::VECCALC>(dacs_brcb2, (shape.BASEL * 64));
        AllocateLocalTensor<TPosition::VECCALC>(cb_buf, (shape.BASEL * shape.BASEL));
        AllocateLocalTensor<TPosition::VECCALC>(da_out, (shape.BASEL * shape.BASEL));
        AllocateLocalTensor<TPosition::VECCALC>(da_out_half, (shape.BASEL * shape.BASEL));
        dacs_buf2.Init(shape.BASEL);
        dtout_buf.Init(shape.BASEL);
        AllocateLocalTensor<TPosition::VECCALC>(maskdiagbuf, (shape.BASEL * shape.BASEL));
        scalea.Init(shape.BASEL);
        AllocateLocalTensor<TPosition::VECCALC>(scalea_brcb, (shape.BASEL * 8));
        AllocateLocalTensor<TPosition::VECCALC>(scalea_brcb2, (shape.BASEL * 64));
        AllocateLocalTensor<TPosition::VECCALC>(states, (shape.BASEL * shape.P));
        out_b.Init((shape.BASEL * shape.P));
        x_half.Init((shape.BASEL * shape.P));
        sumout.Init((shape.BASEL * shape.P));
        // Initialize events
        in_empty.Init();
        in_ready.Init();
        out_empty.Init();
        out_ready.Init();
        tensor_in_empty.Init();
        tensor_in_ready.Init();
        tensor_out_empty.Init();
        tensor_out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        int cnt1 = 0;
        int cnt2 = 0;
        int v1_cnt1 = 0;
        int v1_cnt2 = 0;
        int v1_bc = 0;
        int v1_h = 0;
        LocalTensor<uint32_t> tmptsr_0 = maskdiagbuf.ReinterpretCast<uint32_t>();
        Duplicate<uint32_t, false>(tmptsr_0, (uint32_t)0, MASK_PLACEHOLDER, 64, 1, 8);
        PipeBarrier<PIPE_V>();
        uint64_t mask = 1;
        for (int i=0; i<64; i+=1){
            SetVectorMask<half, MaskMode::NORMAL>(0, mask);
            LocalTensor<uint32_t> tmptsr_1 = maskdiagbuf[(i * 64)].ReinterpretCast<uint32_t>();
            Duplicate<uint32_t, false>(tmptsr_1, (uint32_t)4294967295, MASK_PLACEHOLDER, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            mask = ((mask * 2) + 1);
        }
        SetVectorMask<half, MaskMode::NORMAL>(-1, -1);
        PipeBarrier<PIPE_V>();
        tensor_in_empty.set();
        tensor_out_empty.set();
        VEC_READY(3, PIPE_MTE3);
        VEC_READY(3, PIPE_MTE3);
        for (int bch=shape.BCH1; bch<(shape.BCH2 + 1); bch+=1){
            int bc = ((int)bch / (int)shape.H);
            int h = (bch % shape.H);
            int g = ((int)h / (int)shape.HPERG);
            if ((bch < shape.BCH2)){
                if ((((h % shape.HPERG) == 0) || (bch == shape.BCH1))){
                    WAIT_CUBE(0);
                }
                WAIT_CUBE(3);
                int m1start = 0;
                int m1end = 0;
                int m1 = 0;
                if ((get_subblockid() == 0)){
                    m1start = 3;
                    m1end = 5;
                }
                else {
                    m1start = 1;
                    m1end = 3;
                }
                for (int m1_raw=m1start; m1_raw<m1end; m1_raw+=1){
                    m1 = (m1_raw % 4);
                    for (int m2=0; m2<4; m2+=1){
                        in_empty.wait();
                        tensor_in_empty.wait();
                        if ((m2 == 0)){
                            GM2UB(dacs_buf1.get(cnt1), dacsmtx[(((bc * (shape.H * shape.L)) + (h * shape.L)) + ((m1 * 64) * 1))], 1, ((int)shape.BASEL / (int)8), 0, 0);
                        }
                        if ((m2 <= m1)){
                            GM2UB(dacs_buf2.get(cnt2), dacsmtx[(((bc * (shape.H * shape.L)) + (h * shape.L)) + ((m2 * 64) * 1))], 1, ((int)shape.BASEL / (int)8), 0, 0);
                            GM2UB(dtout_buf.get(cnt2), dtoutmtx[(((bc * (shape.H * shape.L)) + (h * shape.L)) + ((m2 * 64) * 1))], 1, ((int)shape.BASEL / (int)8), 0, 0);
                            GM2UB(cb_buf, cb_ws[(((get_block_idx() * (shape.L * shape.L)) + ((m1 * 64) * shape.L)) + ((m2 * 64) * 1))], shape.BASEL, ((int)shape.BASEL / (int)8), ((int)(shape.L - shape.BASEL) / (int)8), 0);
                        }
                        in_ready.set();
                        tensor_in_ready.set();
                        out_empty.wait();
                        in_ready.wait();
                        tensor_out_empty.wait();
                        tensor_in_ready.wait();
                        if ((m2 == 0)){
                            Brcb(dacs_brcb, dacs_buf1.get(cnt1), ((int)shape.BASEL / (int)8), {1, 8});
                            PipeBarrier<PIPE_V>();
                            Brcb(dacs_brcb2, dacs_brcb, shape.BASEL, {1, 8});
                            PipeBarrier<PIPE_V>();
                        }
                        if ((m2 <= m1)){
                            Sub<float, false>(da_out, dacs_brcb2, dacs_buf2.get(cnt2), MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)64), {1, 1, 1, 8, 8, 0});
                            PipeBarrier<PIPE_V>();
                            Exp<float, false>(da_out, da_out, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)64), {1, 1, 8, 8});
                            PipeBarrier<PIPE_V>();
                            Mul<float, false>(da_out, da_out, cb_buf, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)64), {1, 1, 1, 8, 8, 8});
                            PipeBarrier<PIPE_V>();
                            Mul<float, false>(da_out, da_out, dtout_buf.get(cnt2), MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)64), {1, 1, 1, 8, 8, 0});
                            PipeBarrier<PIPE_V>();
                            if ((m1 == m2)){
                                LocalTensor<uint16_t> tmptsr_2 = da_out.ReinterpretCast<uint16_t>();
                                LocalTensor<uint16_t> tmptsr_3 = da_out.ReinterpretCast<uint16_t>();
                                LocalTensor<uint16_t> tmptsr_4 = maskdiagbuf.ReinterpretCast<uint16_t>();
                                And<uint16_t, false>(tmptsr_2, tmptsr_3, tmptsr_4, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)64), {1, 1, 1, 8, 8, 8});
                                PipeBarrier<PIPE_V>();
                            }
                            Cast<half, float, false>(da_out_half, da_out, RoundMode::CAST_RINT, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)64), {1, 1, 4, 8});
                            PipeBarrier<PIPE_V>();
                        }
                        else {
                            Duplicate<half, false>(da_out_half, (half)0.0, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)128), 1, 8);
                            PipeBarrier<PIPE_V>();
                        }
                        out_ready.set();
                        in_empty.set();
                        tensor_out_ready.set();
                        tensor_in_empty.set();
                        out_ready.wait();
                        tensor_out_ready.wait();
                        UB2GM(mmtx[((((bc * ((shape.H * shape.L) * shape.L)) + (h * (shape.L * shape.L))) + ((m1 * 64) * shape.L)) + ((m2 * 64) * 1))], da_out_half, shape.BASEL, ((int)shape.BASEL / (int)16), 0, ((int)(shape.L - shape.BASEL) / (int)16));
                        out_empty.set();
                        tensor_out_empty.set();
                        cnt2 = (cnt2 + 1);
                    }
                    cnt1 = (cnt1 + 1);
                }
                VEC_READY(0, PIPE_MTE3);
            }
            if ((bch > shape.BCH1)){
                v1_bc = ((int)(bch - 1) / (int)shape.H);
                v1_h = ((bch - 1) % shape.H);
                WAIT_CUBE(1);
                half d_scale_half = 0;
                d_scale_half = (half) dmtx[v1_h].GetValue(0);
                float d_scale = 0;
                d_scale = ((float)d_scale_half);
                int mstart = 0;
                int mend = 0;
                int m = 0;
                if ((get_subblockid() == 0)){
                    mstart = 3;
                    mend = 5;
                }
                else {
                    mstart = 1;
                    mend = 3;
                }
                for (int m2_raw=mstart; m2_raw<mend; m2_raw+=1){
                    m = (m2_raw % 4);
                    in_empty.wait();
                    tensor_in_empty.wait();
                    GM2UB(scalea.get(v1_cnt1), dacsmtx[(((v1_bc * (shape.H * shape.L)) + (v1_h * shape.L)) + ((m * 64) * 1))], 1, ((int)shape.BASEL / (int)8), 0, 0);
                    GM2UB(states, statesmtx[(((v1_bc * ((shape.H * shape.L) * shape.P)) + (v1_h * (shape.L * shape.P))) + ((m * 64) * shape.P))], 1, ((int)(shape.BASEL * shape.P) / (int)8), 0, 0);
                    GM2UB(out_b.get(v1_cnt2), outmtx[(((v1_bc * ((shape.H * shape.L) * shape.P)) + (v1_h * (shape.L * shape.P))) + ((m * 64) * shape.P))], 1, ((int)(shape.BASEL * shape.P) / (int)8), 0, 0);
                    GM2UB(x_half.get(v1_cnt2), xmtx[(((v1_bc * ((shape.L * shape.H) * shape.P)) + ((m * 64) * (shape.H * shape.P))) + (v1_h * shape.P))], shape.BASEL, ((int)shape.P / (int)16), ((int)((shape.H - 1) * shape.P) / (int)16), 0);
                    in_ready.set();
                    tensor_in_ready.set();
                    out_empty.wait();
                    in_ready.wait();
                    tensor_out_empty.wait();
                    tensor_in_ready.wait();
                    Exp<float, false>(scalea.get(v1_cnt1), scalea.get(v1_cnt1), MASK_PLACEHOLDER, ((int)shape.BASEL / (int)64), {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    Brcb(scalea_brcb, scalea.get(v1_cnt1), ((int)shape.BASEL / (int)8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    Brcb(scalea_brcb2, scalea_brcb, shape.BASEL, {1, 8});
                    PipeBarrier<PIPE_V>();
                    Mul<float, false>(states, states, scalea_brcb2, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)64), {1, 1, 1, 8, 8, 8});
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(states, states, out_b.get(v1_cnt2), MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)64), {1, 1, 1, 8, 8, 8});
                    PipeBarrier<PIPE_V>();
                    Cast<float, half, false>(sumout.get(v1_cnt2), x_half.get(v1_cnt2), RoundMode::CAST_NONE, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)64), {1, 1, 8, 4});
                    PipeBarrier<PIPE_V>();
                    Muls<float, false>(sumout.get(v1_cnt2), sumout.get(v1_cnt2), (float)d_scale, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)64), {1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                    Add<float, false>(sumout.get(v1_cnt2), sumout.get(v1_cnt2), states, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)64), {1, 1, 1, 8, 8, 8});
                    PipeBarrier<PIPE_V>();
                    out_ready.set();
                    in_empty.set();
                    tensor_out_ready.set();
                    tensor_in_empty.set();
                    out_ready.wait();
                    tensor_out_ready.wait();
                    UB2GM(sumoutmtx[(((v1_bc * ((shape.L * shape.H) * shape.P)) + ((m * 64) * (shape.H * shape.P))) + (v1_h * shape.P))], sumout.get(v1_cnt2), shape.BASEL, ((int)shape.P / (int)8), 0, ((int)((shape.H - 1) * shape.P) / (int)8));
                    out_empty.set();
                    tensor_out_empty.set();
                    v1_cnt1 = (v1_cnt1 + 1);
                    v1_cnt2 = (v1_cnt2 + 1);
                }
                VEC_READY(3, PIPE_MTE3);
            }
        }
        WAIT_CUBE(3);
        WAIT_CUBE(3);
        tensor_in_empty.wait();
        tensor_out_empty.wait();
        in_empty.release();
        out_empty.release();
    }
    
private:
    CustVecShapeInfo shape;
    // Global Tensors
    GlobalTensor<float> cb_ws;
    GlobalTensor<float> dacsmtx;
    GlobalTensor<float> dtoutmtx;
    GlobalTensor<half> mmtx;
    GlobalTensor<float> outmtx;
    GlobalTensor<float> statesmtx;
    GlobalTensor<half> xmtx;
    GlobalTensor<half> dmtx;
    GlobalTensor<float> sumoutmtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> dacs_buf1;
    LocalTensor<float> dacs_brcb;
    LocalTensor<float> dacs_brcb2;
    LocalTensor<float> cb_buf;
    LocalTensor<float> da_out;
    LocalTensor<half> da_out_half;
    DBuff<float, TPosition::VECCALC> dacs_buf2;
    DBuff<float, TPosition::VECCALC> dtout_buf;
    LocalTensor<float> maskdiagbuf;
    DBuff<float, TPosition::VECCALC> scalea;
    LocalTensor<float> scalea_brcb;
    LocalTensor<float> scalea_brcb2;
    LocalTensor<float> states;
    DBuff<float, TPosition::VECCALC> out_b;
    DBuff<half, TPosition::VECCALC> x_half;
    DBuff<float, TPosition::VECCALC> sumout;
    
    // Events
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
    SEvent<PIPE_V, PIPE_MTE2> tensor_in_empty;
    SEvent<PIPE_MTE2, PIPE_V> tensor_in_ready;
    SEvent<PIPE_MTE3, PIPE_V> tensor_out_empty;
    SEvent<PIPE_V, PIPE_MTE3> tensor_out_ready;
};

} // namespace Mambav2ChunkScan
} // namespace npu_ops_transformer_ext