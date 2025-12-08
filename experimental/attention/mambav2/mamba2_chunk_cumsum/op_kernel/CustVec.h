#pragma once
#include "tensorutils.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkCumsum {
    
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
    shape.nstepsH = (H / 128);
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
        at_buf.Init(64);
        dt_buf.Init(4096);
        dtbias_buf.Init(64);
        dtmask_buf.Init(4096);
        out1buf.Init(4096);
        out2buf.Init(4096);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp0, 4096);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp1, 4096);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp2, 64);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp3, 4096);
        AllocateLocalTensor<TPosition::VECCALC>(cmp_mask, 4096);
        AllocateLocalTensor<TPosition::VECCALC>(max_buf, 8);
        cumsum_tensor.Init(64);
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
        for (int bch=shape.BCH1; bch<shape.BCH2; ++bch){
            int b = (bch / (shape.C * shape.nstepsH));
            int c = ((bch % (shape.C * shape.nstepsH)) / shape.nstepsH);
            int h = (((bch % (shape.C * shape.nstepsH)) % shape.nstepsH) * 128);
            for (int l=0; l<shape.L; l+=64){
                
                in_empty.wait();
                GM2UB(at_buf.get(cc_cnt), at_mtx[(h + ((get_subblockid() * 128) / 2))], 1, 8, 0, 0);
                GM2UB(dt_buf.get(cc_cnt), dt_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + ((get_subblockid() * 128) / 2)))], 64, 4, ((shape.H - 64) / 16), 0);
                GM2UB(dtbias_buf.get(cc_cnt), dtbias_mtx[(h + ((get_subblockid() * 128) / 2))], 1, 4, 0, 0);
                GM2UB(dtmask_buf.get(cc_cnt), dtmask_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + ((get_subblockid() * 128) / 2)))], 64, 4, ((shape.H - 64) / 16), 0);
                in_ready.set();
                
                out_empty.wait();
                in_ready.wait();
                if ((cc_cnt == 0)){
                    Duplicate<float, false>(max_buf, 0.000000f, MASK_PLACEHOLDER, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    Adds<float, false>(max_buf, max_buf, 10000000.0f, MASK_PLACEHOLDER, 1, {0, 0, 0, 0});
                    PipeBarrier<PIPE_V>();
                }
                Cast<float, half, false>(cc_tmp1, dt_buf.get(cc_cnt), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 64, {1, 1, 8, 4});
                Cast<float, half, false>(cc_tmp2, dtbias_buf.get(cc_cnt), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, {1, 1, 8, 4});
                Cast<float, half, false>(cc_tmp3, dtmask_buf.get(cc_cnt), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 64, {1, 1, 8, 4});
                PipeBarrier<PIPE_V>();
                Add<float, false>(cc_tmp1, cc_tmp1, cc_tmp2, MASK_PLACEHOLDER, 64, {1, 1, 1, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                CompareScalar<float, uint8_t>(cmp_mask, cc_tmp1, 20.0f, CMPMODE::LT, 4096);
                PipeBarrier<PIPE_V>();
                UB2UB(cc_tmp0, cc_tmp1, 64, 8, 0, 0);
                PipeBarrier<PIPE_V>();
                Exp<float, false>(cc_tmp1, cc_tmp1, MASK_PLACEHOLDER, 64, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Adds<float, false>(cc_tmp1, cc_tmp1, 1.0f, MASK_PLACEHOLDER, 64, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Ln<float, false>(cc_tmp1, cc_tmp1, MASK_PLACEHOLDER, 64, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Select<float, uint8_t, false>(cc_tmp1, cmp_mask, cc_tmp1, cc_tmp0, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, 64, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                CompareScalar<float, uint8_t>(cmp_mask, cc_tmp1, 10000000.0f, CMPMODE::LT, 4096);
                PipeBarrier<PIPE_V>();
                Select<float, uint8_t>(cc_tmp1, cmp_mask, cc_tmp1, static_cast<float>(10000000.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, 4096);
                PipeBarrier<PIPE_V>();
                Mul<float, false>(out1buf.get(cc_cnt), cc_tmp1, cc_tmp3, MASK_PLACEHOLDER, 64, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                Mul<float, false>(cc_tmp0, out1buf.get(cc_cnt), at_buf.get(cc_cnt), MASK_PLACEHOLDER, 64, {1, 1, 1, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                if ((l > 0)){
                    Add<float, false>(cc_tmp0, cc_tmp0, cumsum_tensor.get(cc_cnt), MASK_PLACEHOLDER, 1, {1, 1, 1, 0, 0, 0});
                    PipeBarrier<PIPE_V>();
                }
                for (int l1=0; l1<63; ++l1){
                    Add<float, false>(cc_tmp0[(((l1 + 1) * 128) / 2)], cc_tmp0[((l1 * 128) / 2)], cc_tmp0[(((l1 + 1) * 128) / 2)], MASK_PLACEHOLDER, 1, {1, 1, 1, 0, 0, 0});
                    PipeBarrier<PIPE_V>();
                }
                if ((l < shape.L)){
                    UB2UB(cumsum_tensor.get((cc_cnt + 1)), cc_tmp0[4032], 1, 8, 0, 0);
                    PipeBarrier<PIPE_V>();
                }
                UB2UB(out2buf.get(cc_cnt), cc_tmp0, 64, 8, 0, 0);
                PipeBarrier<PIPE_V>();
                in_empty.set();
                
                out_ready.set();
                out_ready.wait();
                UB2GM(out_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + ((get_subblockid() * 128) / 2)))], out1buf.get(cc_cnt), 64, 8, 0, ((shape.H - 64) / 8));
                UB2GM(out1_mtx[((((((b * shape.C) + c) * shape.L) * shape.H) + (l * shape.H)) + (h + ((get_subblockid() * 128) / 2)))], out2buf.get(cc_cnt), 64, 8, 0, ((shape.H - 64) / 8));
                if (((l + 64) >= shape.L)){
                    UB2GM(out2_mtx[((((b * shape.C) + c) * shape.H) + (h + ((get_subblockid() * 128) / 2)))], out2buf.get(cc_cnt)[4032], 1, 8, 0, ((shape.H - 64) / 8));
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