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
 * \file sparse_attn_sharedkv_scfa_kernel.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_cube.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_vector.h"
#include "kernel_operator.h"

// 线上编包
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"

// #include "kv_quant_sparse_attn_sharedkv_scfa_common.h"
#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
// using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class KvQuantSparseAttnSharedkvScfa {
public:
    ARGS_TRAITS;
    __aicore__ inline KvQuantSparseAttnSharedkvScfa() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                       __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                       __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
                                       __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                       const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
                                       __gm__ uint8_t *gmTiling, TPipe *tPipe);
    __aicore__ inline void Process();
private:
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const KvQuantSparseAttnSharedkvTilingData *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Process()
{
}

}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H