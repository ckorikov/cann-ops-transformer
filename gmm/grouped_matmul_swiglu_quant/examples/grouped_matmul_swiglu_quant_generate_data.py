#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import sys
import numpy as np
from pathlib import Path

def gen_tensor(low, high, shape, name, dtype=np.float16):
    testCase = sys.argv[1]
    tensor = np.random.uniform(low, high, shape).astype(dtype)
    tensor.tofile(f"{name}.bin")

def generate_non_decreasing_sequence(length, upper_limit):
    """
    生成一个随机非减的一维 Tensor,且最后一个值小于上限, 以满足算子GroupList数值约束。

    参数:
        length (int): 序列的长度。 第二个输入weight 的shape[0]
        upper_limit (int): 最后一个值的上限。第一个输入x 的shape[0]

    返回:
        torch.Tensor: 生成的一维 Tensor。
    """
    # 指定随机种子
    np.random.seed(42)
    # 生成随机递增序列
    random_increments = np.random.randint(0, 128, (length,))  # 随机增量,范围 0~9
    sequence = np.cumsum(random_increments, axis=0)  # 累加生成非减序列

    # 确保最后一个值小于上限
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]  # 计算缩放因子
        sequence = (sequence * scale_factor).astype(np.int64)  # 缩放并转换为整数
    sequence.tofile(f"groupList.bin")

if __name__ == '__main__':
    testCase = sys.argv[1]
    if testCase == 'GMMSwigluQuant_A8W8':
        E, M, K, N = 4, 128, 512, 256
        gen_tensor(-5, 5, (M, K), 'x', dtype=np.int8)
        gen_tensor(-5, 5, (E, N // 32, K // 16, 16, 32), 'weight', dtype=np.int8)
        gen_tensor(-5, 5, (E, N), 'weightScale', dtype=np.float32)
        gen_tensor(-5, 5, (M), 'xScale', dtype=np.float32)
        generate_non_decreasing_sequence(E, M)