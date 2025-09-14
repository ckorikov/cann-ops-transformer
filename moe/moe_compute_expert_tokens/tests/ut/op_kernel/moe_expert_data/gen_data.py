#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import sys
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


def compute_total_rows_before_expert(
        sorted_experts:np.ndarray, 
        num_experts: int) -> np.ndarray:
    arr_length = sorted_experts.shape[-1]
    res = np.arange(num_experts)
    for i in range(num_experts):
        target = i
        low = 0
        high = arr_length - 1
        target_location = -1
        while low <= high:
            mid = (low + high) // 2
            if sorted_experts[mid] > target:
                high = mid - 1
            else:
                low = mid + 1
                target_location = mid
        res[i] = target_location + 1
    return res


def general_input_data(sorted_expert_len, experts_num):
    random_int_list = []
    for _ in range(sorted_expert_len):
        random_int_list.append(random.randint(0, sorted_expert_len))
    sorted_experts = np.sort(random_int_list)
    num_experts = experts_num
    return sorted_experts, num_experts


def gen_golden_data_simple(BSK_len, num_expert):
    sorted_experts, num_experts = general_input_data(BSK_len, num_expert)
    result = compute_total_rows_before_expert(
        sorted_experts, num_experts
    )
    sorted_experts.astype(np.int32).tofile("./sorted_experts.bin")
    np.array(num_experts).astype(np.int32).tofile("./num_experts.bin")
    result.astype(np.int32).tofile("./output_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2])


