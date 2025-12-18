/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"

#define MAX(a,b) (((a)>(b))?(a):(b))
#define DIV_ROUNDUP_MUL(x,y) ((((x)+(y)-1) / (y)) * (y))
#define BYTES_PER_IDX 4 // uint32_t
#define BYTES_PER_VAL 2 // half
#define BYTES_ASCEND_DATA_BLOCK 32
#define DIM0 0
#define DIM1 1
#define DIM2 2
#define DIM3 3
#define DIM128 128

extern void launch_quest_prefill_metadata(
    uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *k_cache,
    uint8_t *block_tables,
    uint8_t *seq_lens,
    uint8_t *metadata_block_tables,
    uint8_t *maxblocks,
    uint8_t *minblocks,
    int32_t B,
    int32_t N,
    int32_t BLOCK_SIZE,
    int32_t D,
    int32_t MKBPR,
    int32_t MMBPR
);


/**
 * This is the interface function which is invoked from the python level. 
 * It handles:
 *  1. resolving tensor shapes
 *  2. Passing the pointers of the input tensors to the kernel
 *  3. Invocation of the kernel
 *  4. Returning the pointer of the output 
 * @brief Interface the `quest prefill metadata` kernel, which initializes of
 *        quest predictor metadata after LLM prefilling of the KV cache
 *
 * @param [in] k_cache (num_kv_blocks, BLOCK_SIZE, N, D)
 * @param [in] block_tables (B, MKBPR) - for every request b: block_tables[b] is 
 *                           a list of kv block indices 
 * @param [in] seq_lens (B,) - sequence length (in token number) per request
 * @param [in] metadata_block_tables (B, MMBPR) - for every request b: block_tables[b] is 
 *                           a list of metadata block indices 
 * @param [out] maxblocks (num_meta_blocks, BLOCK_SIZE, N, D) - maxblock metadata, 
 *                        arranged in blocks of equivalent size to kv_cache blocks
 * @param [out] minblocks (num_meta_blocks, BLOCK_SIZE, N, D) - maxblock metadata, 
 *                        arranged in blocks of equivalent size to kv_cache blocks
 *
 * note: num_kv_blocks can be equal to num_kv_blocks and one may even pass maxblocks = k_cache,
 * minblocks = v_cache to reuse the same page tables of vllm
 */
void quest_prefill_metadata(at::Tensor k_cache,
                            at::Tensor block_tables,
                            at::Tensor seq_lens,
                            at::Tensor metadata_block_tables,
                            at::Tensor maxblocks,
                            at::Tensor minblocks
)
{
    // infer tensor shapes
    int32_t B = seq_lens.sizes()[DIM0]; 
    int32_t N = k_cache.sizes()[DIM2];
    int32_t BLOCK_SIZE = k_cache.sizes()[DIM1];
    int32_t D = k_cache.sizes()[DIM3];
    int32_t MKBPR = block_tables.sizes()[DIM1]; // maximm kv-blocks specifiable per request
    int32_t MMBPR = metadata_block_tables.sizes()[DIM1]; // maximm metadata-blocks specifiable per request

    // validate input shapes
    TORCH_CHECK(D == DIM128, "D must be equal to ", DIM128, " for high performance operations, got ", D);
    TORCH_CHECK(BLOCK_SIZE == DIM128, "BLOCK_SIZE must be equal to ", DIM128, " for high performance operations, got ", BLOCK_SIZE);
    TORCH_CHECK(B == block_tables.size(DIM0), "Batch size mismatch: expected ", B, " from query, got ", block_tables.size(DIM0), " from block_tables");
    TORCH_CHECK(B == metadata_block_tables.size(DIM0), "Batch size mismatch: expected ", B, " from query, got ", metadata_block_tables.size(DIM0), " from metadata_block_tables");
    TORCH_CHECK(N == maxblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", maxblocks.size(DIM2), " from maxblocks");
    TORCH_CHECK(N == minblocks.size(DIM2), "N (num KV heads) mismatch: expected ", N, " from query, got ", minblocks.size(DIM2), " from minblocks");
    TORCH_CHECK(BLOCK_SIZE == maxblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, ", got ", maxblocks.size(DIM1), " from maxblocks");
    TORCH_CHECK(BLOCK_SIZE == minblocks.size(DIM1), "BLOCK_SIZE mismatch: expected ", BLOCK_SIZE, ", got ", minblocks.size(DIM1), " from minblocks");
    TORCH_CHECK(D == maxblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", maxblocks.size(DIM3), " from maxblocks");
    TORCH_CHECK(D == minblocks.size(DIM3), "Head dimension D mismatch: expected ", D, " from query, got ", minblocks.size(DIM3), " from minblocks");

    
    // allocate input tensors
    uint8_t *k_cache_ptr = reinterpret_cast<uint8_t *>(k_cache.storage().data_ptr().get());
    uint8_t *block_tables_ptr = reinterpret_cast<uint8_t *>(block_tables.storage().data_ptr().get());
    uint8_t *seq_lens_ptr = reinterpret_cast<uint8_t *>(seq_lens.storage().data_ptr().get());
    uint8_t *metadata_block_tables_ptr = reinterpret_cast<uint8_t *>(metadata_block_tables.storage().data_ptr().get());
    uint8_t *maxblocks_ptr = reinterpret_cast<uint8_t *>(maxblocks.storage().data_ptr().get());
    uint8_t *minblocks_ptr = reinterpret_cast<uint8_t *>(minblocks.storage().data_ptr().get());

    // set up aunch parameters
    uint32_t blockDims = (B * N > NUM_CORES) ? NUM_CORES : B * N;
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream();

    // launch the kernel
    launch_quest_prefill_metadata(
        blockDims, nullptr, aclStream,
        k_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        metadata_block_tables_ptr,
        maxblocks_ptr,
        minblocks_ptr,
        B,
        N,
        BLOCK_SIZE,
        D,
        MKBPR,
        MMBPR
    );
}

/**
 * Create the binding between this CPP function and python. Expose the function towards python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quest_prefill_metadata", &quest_prefill_metadata,
        R"DOC(
        Interface to the `quest_prefill_metadata` kernel, which initializes
        quest predictor metadata after LLM prefilling of the KV cache. Computes 
        the minimum and maximum metadata vectors for each block in the key 
        cache, enabling efficient sparse attention during the decoding phase.

        Args:
            k_cache (torch.Tensor): Key cache of shape 
                                   [num_kv_blocks, BLOCK_SIZE, N, D] (fp16)
            block_tables (torch.Tensor): KV block tables of shape [B, MKBPR] (int32)
            seq_lens (torch.Tensor): Sequence length of each request in the batch
                                   of shape [B] (int32)
            metadata_block_tables (torch.Tensor): Metadata block tables of 
                                                shape [B, MMBPR] (int32)
            maxblocks (torch.Tensor): Output tensor for maxblock metadata of shape
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16)
            minblocks (torch.Tensor): Output tensor for minblock metadata of shape
                                    [num_meta_blocks, BLOCK_SIZE, N, D] (fp16)

        Note:
            num_kv_blocks can be equal to num_meta_blocks, and one may even pass
            maxblocks = k_cache, minblocks = v_cache to reuse the same page tables
            of vLLM for memory efficiency.
        )DOC",
        py::arg("k_cache"),
        py::arg("block_tables"),
        py::arg("seq_lens"),
        py::arg("metadata_block_tables"),
        py::arg("maxblocks"),
        py::arg("minblocks")
    );
}

