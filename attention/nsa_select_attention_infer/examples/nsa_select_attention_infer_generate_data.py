#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import math
import sys
import torch

np.random.seed(42)
torch.manual_seed(42)
def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_sum, x_max

def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, input_dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=kv_tensor.dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape

def calcu_attention(q_part, k_part, v_part, scale_value, q_dtype):
    qkBmmRes = np.matmul(q_part, k_part.transpose(0, 1, 3, 2), dtype=np.float32)
    qkEleRes = qkBmmRes * scale_value
    softmax_res, softmax_sum, softmax_max = softmax(qkEleRes)
    if q_dtype == np.float16:
        bmm2 = np.matmul(softmax_res.astype(np.float32), v_part.astype(np.float32),
                            dtype=np.float32)
        bmm2Res = bmm2 / softmax_sum
    else:
        bmm2Res = np.matmul(softmax_res.astype(np.bfloat16).astype(np.float32), v_part,
                            dtype=np.float32) / softmax_sum
    return bmm2Res.reshape([1, bmm2Res.shape[1] * bmm2Res.shape[2], 1, bmm2Res.shape[-1]])


case_name = sys.argv[1]
if case_name == 'test_nsa_select_attention_infer':
    q_dtype = np.float16
    q_shape = (1, 1, 1, 192)
    k_shape = (2, 128, 1, 192)
    v_shape = (2, 128, 1, 128)
    topk_indices_shape = (1, 1, 2)
    actual_q_seq_lengths_shape = (1)
    actual_kv_seq_lengths_shape = (1)
    block_table_shape = (1, 2)
    
    scale_value = 1.0
    block_size = 128
    num_heads = 1
    num_key_value_heads = 1
    selected_block_size = 128
    selected_block_count = 2
    q = np.random.uniform(-1, 1, q_shape).astype(q_dtype)
    k = np.random.uniform(-1, 1, k_shape).astype(q_dtype)
    v = np.random.uniform(-1, 1, v_shape).astype(q_dtype)
    topk_indices = np.random.uniform(-1, -1, topk_indices_shape).astype(np.int32)
    atten_mask = None

    actual_q_seq_lengths = np.random.uniform(1, 1, actual_q_seq_lengths_shape).astype(np.int64)
    actual_kv_seq_lengths = np.random.uniform(256, 256, actual_kv_seq_lengths_shape).astype(np.int64)

    batch = q_shape[0]
    numHeads = num_heads
    numKeyValueHeads = num_key_value_heads
    headDim = k_shape[3]
    headDimV = v_shape[3]
    blockNum = block_table_shape[0] * block_table_shape[1]
    blockNumPerBlock = []
    block_num_min = 0
    for batch_index in range(batch):
        act_seqlen = actual_kv_seq_lengths[batch_index]
        blockNumPerBlock.append(math.ceil(act_seqlen / block_size))
        block_num_min += math.ceil(act_seqlen / block_size)
    if block_num_min > blockNum:
        print(f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
        exit(1)
    block_idx_list = np.arange(0, blockNum, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
    block_idx = 0
    block_table = [-1] * block_table_shape[1]
    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in blockNumPerBlock:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    max_act_seqlen = -1
    act_seqlen_topk_list = []
    for batch_idx in range(batch):
        act_seqlen = actual_kv_seq_lengths[batch_idx]
        max_act_seqlen = max(act_seqlen, max_act_seqlen)
        # 计算当前batch的有效块数
        valid_blocks_max = math.ceil(act_seqlen / selected_block_size)
        act_seqlen_tail = act_seqlen - (valid_blocks_max - 1) * selected_block_size
        valid_blocks_topk = np.random.randint(0, valid_blocks_max)
        if valid_blocks_topk == 0:
            valid_blocks_topk = 1
        act_seqlen_topk = valid_blocks_topk * selected_block_size if (valid_blocks_topk < valid_blocks_max) else max_act_seqlen
        block_indices = torch.randperm(valid_blocks_max).numpy().astype(np.int32)
        for topk_idx in range(valid_blocks_topk):
            if block_indices[topk_idx] == (valid_blocks_max - 1):
                act_seqlen_topk = (valid_blocks_topk - 1) * selected_block_size + act_seqlen_tail
                break
        act_seqlen_topk_list.append(act_seqlen_topk)
        for numHeads_idx in range(numKeyValueHeads):
            topk_indices[batch_idx, numHeads_idx, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
            for vaild_idx in range(valid_blocks_topk, selected_block_count):
                topk_indices[batch_idx, numHeads_idx, vaild_idx] = -1
    k_cache = np.zeros([batch, max_act_seqlen, numKeyValueHeads, headDim])
    v_cache = np.zeros([batch, max_act_seqlen, numKeyValueHeads, headDimV])
    for batch_idx in range(batch):
        act_seqlen = actual_kv_seq_lengths[batch_idx]
        act_seqlen_topk = act_seqlen_topk_list[batch_idx]
        block_table_cur = block_table[batch_idx]
        for numHeadsIdx in range(numKeyValueHeads):
            idIntopKRecord = -1
            x = 0
            currentTopDeal = -1
            topKIdOffset = 0
            nCopyRowCount = 128
            nActCopyRowCount = 128
            copyRowTimes = math.ceil(act_seqlen_topk / nCopyRowCount)
            for_loop_break = False
            for copyRowIdx in range(copyRowTimes):
                if for_loop_break:
                    break
                copyFinishRowCnt = 0
                curSeqIdx = 128 * copyRowIdx
                if copyRowIdx == copyRowTimes-1:
                    nActCopyRowCount = act_seqlen_topk - nCopyRowCount * (copyRowTimes-1)
                while (copyFinishRowCnt < nActCopyRowCount):
                    if x == currentTopDeal:
                        topKIdOffset += 1
                    idIntopk = topk_indices[batch_idx, numHeadsIdx, topKIdOffset]
                    if idIntopk == -1:
                        for_loop_break = True
                        break
                    else:
                        if (idIntopKRecord != idIntopk):
                            x = 0
                            idIntopKRecord = idIntopk
                            global_start = idIntopk * selected_block_size
                            global_end = act_seqlen if (global_start + selected_block_size > act_seqlen) else (global_start + selected_block_size)
                            currentTopDeal = global_end - global_start
                        global_start = idIntopk * selected_block_size
                        global_end = act_seqlen if (global_start + selected_block_size > act_seqlen) else (global_start + selected_block_size)
                        global_start += x
                        start_offset = global_start % block_size
                        start_block_idx = global_start // block_size
                        end_block_idx = (global_end - 1) // block_size
                        reaminRowCnt = start_offset
                        copyRowCnt = global_end - global_start if (start_block_idx == end_block_idx) else (block_size - reaminRowCnt)
                        if (copyFinishRowCnt + copyRowCnt > nActCopyRowCount):
                            copyRowCnt = nActCopyRowCount - copyFinishRowCnt
                        block_idx = block_table_cur[start_block_idx]
                        k_cache[batch_idx, curSeqIdx : curSeqIdx + copyRowCnt, numHeadsIdx, :] = k[block_idx, start_offset : (start_offset + copyRowCnt), numHeadsIdx,:]
                        v_cache[batch_idx, curSeqIdx : curSeqIdx + copyRowCnt, numHeadsIdx, :] = v[block_idx, start_offset : (start_offset + copyRowCnt), numHeadsIdx,:]
                        copyFinishRowCnt += copyRowCnt
                        x += copyRowCnt
                        curSeqIdx += copyRowCnt

    group = numHeads // numKeyValueHeads
    y = np.zeros([batch, numHeads, 1, v.shape[-1]])
    q = q.transpose(0, 2, 1, 3)
    k_cache = k_cache.transpose(0, 2, 1, 3)
    v_cache = v_cache.transpose(0, 2, 1, 3)
    k_cache, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_cache, q_dtype)

    v_cache, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_cache, q_dtype)
    for batch_index in range(batch):
        act_seqlen = act_seqlen_topk_list[batch_index]
        y[batch_index:(batch_index+1), :, :, :] = calcu_attention(q[batch_index:(batch_index+1), :, :, :], k_cache[batch_index:(batch_index+1), :, 0:act_seqlen, :], v_cache[batch_index:(batch_index+1), :, 0:act_seqlen, :], scale_value, q_dtype)
    attention_out = y.reshape(batch, 1, numHeads, v.shape[-1])
    if (q_dtype == np.float16):
        attention_out = attention_out.astype(np.float16)
    else:
        attention_out = attention_out.astype(np.float32)
    
    q = q.transpose(0, 2, 1, 3)

    q.tofile("query.bin")
    k.tofile("key.bin")
    v.tofile("value.bin")
    topk_indices.tofile("topk_index.bin")
    actual_q_seq_lengths.tofile("actual_q_seq_lengths.bin")
    actual_kv_seq_lengths.tofile("actual_kv_seq_lengths.bin")
    block_table.tofile("block_table.bin")
    attention_out.tofile("attention_out.bin")
else:
    raise RuntimeError(f"Invalid case name:", case_name)