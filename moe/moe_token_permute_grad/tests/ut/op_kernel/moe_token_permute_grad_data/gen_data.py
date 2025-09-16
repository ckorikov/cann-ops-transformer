#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

import sys
import numpy as np
import torch


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    top_k,
    probs: torch.Tensor = None
):
    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = top_k

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens




def get_cpu_data(permuted_tokens: torch.Tensor, sorted_indices: torch.Tensor, top_k, probs: torch.Tensor = None):
    permuted_tokens = permuted_tokens.float()
    if probs is not None:
        probs = probs.float()
    unpermuted_tokens = unpermute(permuted_tokens, sorted_indices, top_k, probs)
    return unpermuted_tokens.bfloat16()

def gen_data_and_golden(tokens_num, topk, hidden_size, haveProbs): # flag=True传probs, False不传probs
    permuted_tokens = torch.rand(tokens_num*topk, hidden_size)
    sorted_indices_golden = torch.randint(low=0,high=tokens_num*topk,size=(tokens_num*topk)) # 传给标杆
    if haveProbs:
        probs = torch.rand(tokens_num,topk)
    else:
        probs = None
    unpermuted_output = get_cpu_data(permuted_tokens, sorted_indices_golden, topk, probs)

    permuted_tokens = permuted_tokens.numpy()
    sorted_indices = torch.argsort(sorted_indices_golden, stable=True).to(torch.int64).numpy() # 传给算子
    unpermuted_output = unpermuted_output.numpy()

    permuted_tokens.tofile("./input_permuted_tokens.bin")
    sorted_indices.tofile("./input_sorted_indices.bin")
    if haveProbs:
        probs = probs.numpy()
        probs.tofile("./input_probs.bin")
    unpermuted_output.tofile("./output_unpermuted_tokens.bin")



if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])