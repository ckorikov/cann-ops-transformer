import math
import random
import os
from datetime import datetime
import numpy as np 
import pandas as pd 
import pytest 
import torch
import torch.nn.functional as F 
import torch_npu
import random
import test_V1

def validate_config(batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, in_layout):
    # 校验
    if batch_size > 65536:
        raise ValueError("batch_size must <= 65536")
    if q_head_num > 256:
        raise ValueError("q_head_num must <= 256")
    if kv_head_num > 256:
        raise ValueError("kv_head_num must <= 256")
    if q_head_num % kv_head_num:
        raise ValueError("q_head_num should be intergral multiple of kv_head_num")
    if head_dim > 512:
        raise ValueError("head_dim must <= 512")
    if dtype not in [torch.bfloat16, torch.float16, torch.int8]:
        raise ValueError("dtype should be: float16/bfloat16/int8")
    if in_layout not in ["BSH", "BSND", "BNSD"]:
        raise ValueError("input_layout should be: BSH/BSND/BNSD")

def load_excel_test_cases():
    # 加载测试用例
    filename = os.getenv('EXCEL_FILE')
    sheetname = os.getenv('SHEET_NAME', 'IFA_FIA_Case') # 优先使用环境变量，如果没有则使用默认值

    if not filename:
        pytest.skip("\nset EXCEL_FILE=xxx.xlsx(file path), eg: EXCLE_FILE=file_path python3 -m pytest -rA -s test_excel.py", allow_module_level=True)

    if not os.path.exists(filename):
        pytest.skip(f"Excel file: {filename} not exist!", allow_module_level=True)
    
    try:
        df = pd.read_excel(filename, sheetname=sheetname)
        print(f"\nSuccessfully reading excel file: {filename}")
        print(f"Sheet: {sheetname}")
        print(f"number of testcases: {len(df)}")

        test_cases = []
        for index, row in df.iterrows():
            test_cases.append((row['Testcase_Name'], row['inputLayout'], row['q_shape'], row['q_dtype'], \
                row['k_shape'], row['k_cache_shape'], row['actual_seq_lengths_kv'], row['blockSize'], row['scaleValue'], row['kn_pre'], row['kn_nxt']))

        return test_cases
    
    except Exception as e:
        pytest.skip(f"failed to read excel file: {e}", allow_module_level=True)


def convert_excel_pytest(inputLayout, q_shape, q_dtype, k_shape, k_cache_shape, actual_seq_lengths_kv, blockSize, scaleValue, kn_pre, kn_nxt):
    # 解析形状
    q_shape_list = [int(x.strip()) for x in q_shape.split(',')]
    kv_shape_list = [int(x.strip()) for x in k_shape.split(',')]
    kv_cache_shape_list = [int(x.strip()) for x in k_cache_shape.split(',')]
    kv_actual_seq_list = [int(x.strip()) for x in actual_seq_lengths_kv.split(',')]

    if inputLayout == "BNSD":
        batch_size, q_head_num, q_seq, head_dim = q_shape_list
        _, kv_head_num, kv_seq, _ = kv_shape_list
    elif inputLayout == "BSND":
        batch_size, q_seq, q_head_num, head_dim = q_shape_list
        _, kv_seq, kv_head_num, _ = kv_shape_list
    else:
        raise ValueError("Right now only support: BNSD, BSND")
    
    if q_dtype == "BF16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    scaled_value = float(scaleValue)

    if len(kv_cache_shape_list) == 3:
        cache_layout = "BBH"
        _, block_size, _ = kv_cache_shape_list
    else:
        cache_layout = "BNBD"
        _, _, block_size, _ = kv_cache_shape_list

    act_seq_len = [q_seq] * batch_size
    return batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, act_seq_len, kv_actual_seq_list, block_size, cache_layout, scaled_value


    def creat_random_block_table(batch_size, act_seq_len_kv, block_size):
        # 计算最大序列长度
        max_kv_seq_length = max(act_seq_len_kv)

        # 计算每个序列实际需要的block数量
        valid_block_num = [math.ceil(act_seq_len_kv[i] / block_size) for i in range(batch_size)]
        valid_block_num_sum = sum(valid_block_num)

        # 计算最大block数量
        max_valid_block_num = math.ceil(max_kv_seq_length / block_size)

        # 随机分配block table
        block_table = -torch.ones(batch_size, max_valid_block_num, dtype=torch.int32)
        block_idx = list(range(valid_block_num_sum))
        random.shuffle(block_idx)
        block_idx_start = 0

        for i in range( batch_size):
            block_table[i][: valid_block_num[i]] = torch.tensor(
                block_idx[block_idx_start : block_idx_start + valid_block_num[i]],
                dtype=torch.int32
            )
            block_idx_start += valid_block_num[i]

        return block_table, valid_block_num_sum


def check_result(expect, result):
    result_cpu = result.cpu()
    expect_cpu = result.cpu()
    attn_out_diff = result_cpu.reshape(-1) - expect_cpu.reshape(-1)
    thres = 0.005
    idx = torch.nonzero(abs(attn_out_diff) > thres).squeeze()
    values = attn_out_diff[abs(attn_out_diff) > thres]
    print(f'\ndiff大于{thres}的元素索引:,{idx}')
    print(f'diff大于{thres}的元素值:,{values}')
    print(f'pass num:{torch.sum(abs(attn_out_diff.squeeze()) <= thres)}, total num:{attn_out_diff.numel()}, accur: {torch.sum(abs(attn_out_diff.squeeze()) <= thres) / attn_out_diff.numel()}\n\n')

    assert torch.allclose(result_cpu.flatten(), expect_cpu.flatten(), rtol=0.05, atol=0.05)