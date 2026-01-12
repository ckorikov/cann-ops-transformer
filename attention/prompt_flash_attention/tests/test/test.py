# -*- coding: utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
import numpy as np
import torch

import time
import math
import random

import ctypes
import copy
import math
from ml_dtypes import float8_e5m2
from ml_dtypes import float8_e4m3fn
import pandas as pd
random.seed(1)
np.random.seed(1)
torch.manual_seed(11)

def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result

def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print(
        'Error Line-----------------------------------------------------------------------------')
    print('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print(
        '---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        # print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
        # i, expect_data[i], real_data[i], abs(np.float64(
        #     expect_data[i]) - np.float64(real_data[i])),
        # relative_diff[count - 1]))
        if count < 10 or (90 < count < 100):
            print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(
                    expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print('%08s \t %07s \t %07s \t %07s \t %07s' %
                      (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print(
        'Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) -
                    np.float64(real_data[m_idx])),
                max_error))
        else:
            break
    print(
        '---------------------------------------------------------------------------------------')

def display_output(real_data, expect_data, start, end, diff_thd, expect_fp32_data=None, if_mix=False):
    def display_inner(idx):
        j = idx + start
        if if_mix:
            diff_rate = cal_relative_diff_mix(
                expect_data[j], real_data[j], diff_thd)
        else:

            diff_rate = cal_relative_diff(
                expect_data[j], real_data[j], diff_thd)

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(
                expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))

    print(
        '---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        print(
            'Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        print('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print(
        '---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)

def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result

def cal_nan_inf_diff(real_data, expect_data, output_dtype, diff_abs, diff_thd, params):
    err_diff = []
    err_idx = []
    aclnn_name = params["op_name"]

    if output_dtype == 'fp32':
        inf_value = 3.4028e38
    elif output_dtype == 'bf16':
        inf_value = 3.38e38
    else:
        inf_value = 65504

    real_data_copy = copy.deepcopy(real_data)
    expect_data_copy = copy.deepcopy(expect_data)
    inf_idx = np.where(np.isinf(real_data_copy))[0]
    pos_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] > 0)[0]
    neg_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] < 0)[0]
    real_data_copy[inf_idx[pos_inf_idx]] = inf_value
    real_data_copy[inf_idx[neg_inf_idx]] = -inf_value
    inf_idx = np.where(np.isinf(expect_data_copy))[0]
    pos_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] > 0)[0]
    neg_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] < 0)[0]
    expect_data_copy[inf_idx[pos_inf_idx]] = inf_value
    expect_data_copy[inf_idx[neg_inf_idx]] = -inf_value

    num_idx = np.where(~np.isnan(real_data_copy) & ~np.isinf(real_data_copy) & ~np.isnan(expect_data_copy) & ~np.isinf(
        expect_data_copy))
    nan_inf_idx = np.setdiff1d(np.arange(len(real_data_copy)), num_idx)

    rdiff = cal_relative_diff_np(real_data_copy[num_idx].astype(np.float32),
                                 expect_data_copy[num_idx].astype(np.float32),
                                 diff_thd)
    num_err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = num_idx[0]
    num_err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

    real_data_str = list(map(lambda x: str(x), real_data[nan_inf_idx].tolist()))
    expect_data_str = list(map(lambda x: str(x), expect_data[nan_inf_idx].tolist()))

    temp_err_idx = np.where(np.array(real_data_str) != np.array(expect_data_str))[0]
    nan_inf_err_idx = nan_inf_idx[temp_err_idx]
    nan_inf_err_diff = diff_abs[nan_inf_err_idx]

    err_idx = num_err_idx.tolist() + nan_inf_err_idx.tolist()
    err_diff = num_err_diff.tolist() + nan_inf_err_diff.tolist()

    return np.array(err_diff), np.array(err_idx)

def data_compare_np(params, npu_output, cpu_output, output_dtype, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1,
                    precision_method=0,
                    rtol=0.005, atol=0.000025):
    max_error_idx = 10000000
    print("[data_compare_np]ouput flatten start!")
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    print("[data_compare_np]ouput flatten end!")
    if real_data.size == 0 and real_data.size == data_compe.size:
        print(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        print(
            'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(
        data_compe)].size + data_compe[np.isnan(data_compe)].size
    print("[data_compare_np]overflows_count start!")
    if overflows_count > 0:
        print('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))
    print("[data_compare_np]overflows_count end!")

    split_count = int(end - start + 1) if end != start else 1
    print('split_count:%s; max_diff_hd:%s;' %
              (float(split_count), max_diff_hd))

    has_nan_inf = False
    if 'nan' in str(real_data) or 'inf' in str(real_data) or 'nan' in str(data_compe) or 'inf' in str(data_compe):
        has_nan_inf = True

    # 默认精度比对方式
    if precision_method == 0:
        print("[data_compare_np]diff_abs start!")
        try:
            diff_abs = np.abs(np.subtract(real_data.astype(
                np.float32), data_compe.astype(np.float32)))
        except MemoryError:
            return result, 0.0, max_error
        print("[data_compare_np]diff_abs end!")
        if has_nan_inf:
            err_diff, err_idx = cal_nan_inf_diff(real_data, data_compe, output_dtype, diff_abs, diff_thd, params)
        else:
            diff_index = np.where(diff_abs > 0)
            rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                         data_compe[diff_index].astype(np.float32),
                                         diff_thd)
            err_diff = rdiff[rdiff > diff_thd]
            diff_idx_list = diff_index[0]
            err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

        fulfill_percent = float(split_count - err_diff.size) / \
                          float(split_count) * 100.0

        display_output(real_data, data_compe, start, end, diff_thd)
        pct_thd = (1 - pct_thd) * 100.0
        result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
        if len(err_diff) > 0:
            max_error = max(err_diff[0:max_error_idx])
            if max_error >= max_diff_hd:
                result = "Failed"
        print(
            '---------------------------------------------------------------------------------------')
        print('DiffThd  \t PctThd   \t PctRlt   \t Result')
        print(
            '---------------------------------------------------------------------------------------')
        print('%.4f     \t %.2f%%   \t %.6f%%   \t %s' %
                  (diff_thd, pct_thd, fulfill_percent, result))
        if len(err_diff) > 0:
            print('Max-RelativeError is: %s. Threshold is: %s.' %
                      (max_error, max_diff_hd))
        if result == "Failed":
            display_error_output(real_data, data_compe,
                                 err_idx, err_diff[0:max_error_idx])


def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def _random_fill_tensor(tensor, shape, random_number, value=0):
    for i in range(0, random_number):
        point = []
        for k in range(0, len(shape)):
            point.append(random.randint(1, shape[k]) - 1)
        tensor[point[0], point[1]] = value
    return tensor


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D])
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape

def _np_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor,dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape

def broadcast_kv_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    factor = numheads // numKeyValueHeads
    shape = tensor.shape
    D = shape[3]
    if 'bfloat16' in str(dtype):
        tensor_new = np.zeros([2, numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
    else:
        tensor_new = np.zeros([2, numheads, 1, D], dtype=dtype)
    for i in range(numheads):
        j = i // factor
        tensor_new[:, i:i + 1, :, :] = tensor[:, j:j + 1, :, :]
    return tensor_new

def broadcast_kv_split_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    factor = numheads // numKeyValueHeads
    shape = tensor.shape
    D = shape[2]
    if 'bfloat16' in str(dtype):
        tensor_new = np.zeros([numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
    else:
        tensor_new = np.zeros([numheads, 1, D], dtype=dtype)
    for i in range(numheads):
        j = i // factor
        tensor_new[i:i + 1, :, :] = tensor[j:j + 1, :, :]
    return tensor_new

def _create_mask_right_down(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,prefix_kvs, batch, numheads, kvs_list,m_dtype):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i]+actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i]+actualprefixKV
        else:
            S2 = mask_s_kv-prefix_kvs+actualprefixKV
        next_tokens = S2 - S1
        next_tokens_list.append(next_tokens)
        atten_masks = _create_mask(m_shape, pre_tokens, next_tokens)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, next_tokens_list

def _create_mask(m_shape, pre_tokens, next_tokens):
    next_masks = np.triu(np.ones(m_shape, dtype='uint8'), k=1 + int(next_tokens))  # 生成下三角全是0的矩阵
    pre_mask = np.tril(np.ones(m_shape, dtype='uint8'), k=-1 - int(pre_tokens))  # 生成上三角全是0的矩阵
    atten_masks = pre_mask + next_masks

    return atten_masks
def _create_mask_no_sparse(m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
    re_mask_batch = []
    re_mask_npu_batch = []

    pad_flag = False
    npu_mask = None
    if m_shape[0] !=npu_m_shape[0] or m_shape[1]!=npu_m_shape[1]:
        pad_flag = True
        npu_mask = np.ones(npu_m_shape, dtype='uint8')
    cpu_mask = _create_mask(m_shape, pre_tokens, next_tokens)
    if pad_flag:
        if batch == None:
            cpu_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            npu_mask[:cpu_mask.shape[0], :cpu_mask.shape[1]] = cpu_mask
            return cpu_mask, npu_mask
        for i in range(batch):
            re_mask_num = []
            re_mask_npu_num = []
            re_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)

            npu_mask[:re_mask.shape[0], :re_mask.shape[1]] = re_mask
            if numheads:
                # cpu
                re_mask_num.append(re_mask)
                re_mask_batch.append(re_mask_num)
                # npu

                re_mask_npu_num.append(npu_mask)
                re_mask_npu_batch.append(re_mask_npu_num)
            else:
                re_mask_batch.append(re_mask)
                re_mask_npu_batch.append(npu_mask)
        cpu_mask = np.array(re_mask_batch).astype(m_dtype)
        npu_mask = np.array(re_mask_npu_batch).astype(m_dtype)
        return cpu_mask, npu_mask
    else:
        if batch ==None:
            cpu_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            return cpu_mask,cpu_mask
        for i in range(batch):
            re_mask_num = []
            re_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            if numheads:
                re_mask_num.append(re_mask)
                re_mask_batch.append(re_mask_num)
            else:
                re_mask_batch.append(re_mask)

        cpu_mask = np.array(re_mask_batch).astype(m_dtype)
        return cpu_mask,cpu_mask
def _create_mask_left_up(m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
    re_mask_batch = []
    attentionmask = _create_mask(m_shape, pre_tokens, next_tokens)
    for i in range(batch):
        re_mask_batch.append(attentionmask)
    return re_mask_batch

def _create_mask_band(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV,actualprefixKV, prefix_kvs,batch, numheads, kvs_list, m_dtype):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    pre_tokens_list = []
    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i]+actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i]+actualprefixKV
        else:
            S2 = mask_s_kv-prefix_kvs+actualprefixKV
        pre_tokens_new = S1-S2+pre_tokens
        pre_tokens_list.append(pre_tokens_new)

        next_tokens_new = S2-S1+next_tokens
        next_tokens_list.append(next_tokens_new)
        atten_masks = _create_mask(m_shape, pre_tokens_new, next_tokens_new)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, pre_tokens_list, next_tokens_list
def _t_gpu_create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch=1, numheads=1, sp_mode=0, random_ones=0):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    print(f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}")
    cpu_mask = None
    if sp_mode == 0:
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype, random_ones)
    if sp_mode == 1:
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype, random_ones)
    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype)
    if sp_mode == 3:
        pre_tokens = 214748647
        next_tokens = 0
        print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
        cpu_mask, nexttokens = _create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch, numheads, m_dtype)
    if sp_mode ==4:
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype)
    cpu_mask = torch.from_numpy(cpu_mask)
    return cpu_mask.to(m_dtype)



def _create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,prefix_kvs,kvs_list,batch=1, numheads=1, sp_mode=0, random_ones=0):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    print(
        f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}")
    if sp_mode == 0:
        cpu_mask, npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,
                                                    m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 1:
        print(f"[_create_random_mask_by_spars] sp_mode is 1 return all zero mask")
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask, npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,
                                                    m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens



    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 3:  # rightdown
        pre_tokens = 214748647
        print(f"[_create_random_mask_by_spars] sp_mode is 3 npu mask shape:{npu_m_shape}")
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, next_tokens_new = _create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths,
                                                            actualSeqLengthsKV, actualprefixKV, prefix_kvs, batch, numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens_new
    if sp_mode == 4:
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, pre_tokens_new, next_tokens_new = _create_mask_band(cpu_m_shape, pre_tokens, next_tokens,
                                                                      actualSeqLengths, actualSeqLengthsKV, actualprefixKV, prefix_kvs,batch,
                                                                      numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new
def t_bsh_to_bsnd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
        gpu golden  need  bsnd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        if B == 1:
            return tensor.reshape(B, S, N, D), [B, S, N, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    print(sums + j)
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp, [B, S, N, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D), [B, S, N, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D).permute(0, 2, 1, 3), [B, S, N, D]
    elif inputLayout == "TND":
        N = bsh_shape[1]
        D = bsh_shape[2]
        B = len(actSeqLength)
        S = max(actSeqLength)
        new_tensor = np.zeros((B, S, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for s_index in range(act_s):
                new_tensor[b_index, s_index, 0:N, :] = tensor[t_start:t_end, :, :]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, S, N, D]
    else:
        print("BNSD", tensor.shape)
        return tensor.permute(0, 2, 1, 3), bsh_shape

def _t_trans_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
        gpu_small golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp.permute(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "TND":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print("BNSD", tensor.shape)
        return tensor, bsh_shape

def trans_bnsd_to_bsh(tensor, shape):
    B = shape[0]
    N = shape[1]
    S = shape[2]
    D = shape[3]
    H = N * D
    return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)


def np_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
    cpu golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = np.zeros((B, S, N, D), dtype=tensor.dtype)
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "BSND":
        print("BSND")
        B = bsh_shape[0]
        S = bsh_shape[1]
        N = bsh_shape[2]
        D = bsh_shape[3]
        return tensor.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "TND":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print(inputLayout)
        return tensor, bsh_shape
def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m
def alibi_biases(pre_shift_shape):
    # [0,  1, 2]
    # [-1, 0, 1]
    # [-2,-1, 0]
    q_seq = pre_shift_shape[2]
    kv_seq = pre_shift_shape[3]
    alibi_biases = np.zeros([q_seq, kv_seq])
    for x in range(0, kv_seq):
        for y in range(0, q_seq):
            alibi_biases[y, x] = abs(x - y - (kv_seq - q_seq))*(-1)
    return alibi_biases

def get_all_alibi(numHeads,pre_shift_shape,pse_dtype=np.float32): #输入是bnss或者是1nss B, Nq, Sq, Skv, pse_layout, pse_type, pse_dtype,pre_shift_shape
    m = get_slopes(numHeads)  #m  系数
    m = m.numpy()
    alibi_biase = alibi_biases(pre_shift_shape)
    pse_shift = np.zeros(pre_shift_shape,dtype=np.float32)

    for n in range(numHeads):
        pse_shift[:, n:n + 1, :, :] = alibi_biase * m[n]
    return pse_shift.astype(pse_dtype)

def _np_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    print(f"broadcast_mask_n:mask shape:{pse_shift_shape} q_batch:{q_batch}")
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        # pse_res = np.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        pse_res = np.zeros([B, pse_shift_tensor.shape[1], pse_shift_tensor.shape[2], pse_shift_tensor.shape[3]])
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res
def _t_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    print(f"broadcast_mask_n:mask shape:{pse_shift_shape} q_batch:{q_batch}")
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        pse_res = torch.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res
def _np_broadcast_mask_n(m_tensor, m_shape,cpu_m_shape, numheads, q_batch):
    print(f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}")
    mask_cur_shape = cpu_m_shape
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[2], m_shape[3]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = []
        for i in range(B):
            m_res.append(m_tensor)
        return m_res
    else:
        return m_tensor

def _t_broadcast_mask_n(m_tensor, m_shape, numheads, q_batch):
    print(f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}")
    if len(m_shape) == 4:
        # b1ss
        B = m_shape[0]

        m_res = torch.zeros([B, numheads, m_shape[2], m_shape[3]])
        for i in range(B):
            mask_cur_shape = [m_shape[2], m_shape[3]]
            mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = mask_cur
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = torch.zeros([B, numheads, m_shape[1], m_shape[2]])
        for i in range(B):
            mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = mask_cur
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = torch.zeros([B, numheads, m_shape[0], m_shape[1]])
        for i in range(B):
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor
        return m_res
    else:
        return m_tensor


def npSoftmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum

def npSoftmax_new(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_max, x_sum


def _t_softmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def softmax_flashv2(x, max_front=None, sum_front=None, update=None, is_fp16=False):
    """
    Compute the softmax function for each channel of the input x.
    """
    if update == None:
        if is_fp16:
            x = x.astype(np.float32)
        x_max = np.max(x, axis=-1, keepdims=True)
        # print("#########"*4)
        # print(x_max)
        # print(x)
        x_sub = x - x_max  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        out = x_exp
        # out = x_exp / x_sum
        exp_max = None
        if is_fp16:
            out = out.astype(np.float16)
            # x_max = x_max.astype(np.float16)
            # x_sum = x_sum.astype(np.float16)

        return out, x_max, x_sum, exp_max
    else:
        if is_fp16:
            x = x.astype(np.float32)
            max_front = max_front.astype(np.float32)
            sum_front = sum_front.astype(np.float32)
        x_max_tmp = np.max(x, axis=-1, keepdims=True)  # tmp
        x_sub = x - x_max_tmp  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)  # tmp
        x_max = np.max(np.concatenate((max_front, x_max_tmp), axis=-1), axis=-1, keepdims=True)  # ->x_max
        x_exp_new = np.exp(x_max_tmp - x_max)  # -> x_max_tmp
        exp_max = np.exp(max_front - x_max)  # -> exp_max
        # update sum
        exp_max = exp_max * sum_front
        reduce_tmp = x_exp_new * x_sum
        x_sum = exp_max + reduce_tmp  # x_sum

        # exp_max = exp_max/x_sum
        exp_max = np.exp(max_front - x_max)
        out = x_exp * x_exp_new
        # out = x_exp*x_exp_new / x_sum

        # ### softmax ratio
        # softma_ratio = x_sum * x_exp_new / x_sum
        # out_new = out * softma_ratio
        # pround("out_new===================", out_new)
        if is_fp16:
            out = out.astype(np.float16)
            # x_max = x_max.astype(np.float16)
            # x_sum = x_sum.astype(np.float16)
            # exp_max = exp_max.astype(np.float16)

        return out, x_max, x_sum, exp_max


def _np_pfaattention_act_int8(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    S = None
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']


    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end-kvs_begin


    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("start matmul1")
        qk_i = np.matmul(q_tensor.astype(np.int32), ki.transpose(0, 1, 3, 2).astype(np.int32))

        # print("dequant_scale1", dequant_scale1)
        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        print("end matmul1")

        # 转成fp16防止溢出
        # qk_i = np.where(qk_i > 65504, 65504, qk_i)
        # qk_i = np.where(qk_i < -65504, -65504, qk_i)
        # qk_i = qk_i.astype(np.float16)
        qk_i = qk_i* np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=True)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            # 转成int8防止溢出
            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=True)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            # 转成int8防止溢出
            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            # 转成fp16防止溢出
            # o_i = np.where(o_i > 65504, 65504, o_i)
            # o_i = np.where(o_i < -65504, -65504, o_i)
            o_i = o_i.astype(np.float32)

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front


    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)


    return o_front,lse

def _np_pfaattention_act_fp8(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    S = None
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']


    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end-kvs_begin


    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("qDtype:", qDtype)
        print("start matmul1")
        qk_i = np.matmul(q_tensor.astype(np.float32), ki.transpose(0, 1, 3, 2).astype(np.float32))
        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        # print(qk_i)
        print("end matmul1")

        qk_i = qk_i* np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=False)
            # print(py_s_i)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            if qDtype == "float8_e5m2":
                py_s_i = py_s_i.astype(float8_e5m2)
            else:
                py_s_i = py_s_i.astype(float8_e4m3fn)
            # print(py_s_i)

            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            if qDtype == "float8_e5m2":
                py_s_i = py_s_i.astype(float8_e5m2)
            else:
                py_s_i = py_s_i.astype(float8_e4m3fn)

            print(py_s_i.dtype)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front


    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)


    return o_front,lse

def _np_pfaattention_act_hifp8(q_tensor_hifp8, k_tensor_hifp8, v_tensor_hifp8, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    S = None
    q_tensor = trans_np_hifuint8_tensor_to_float32(q_tensor_hifp8).astype(np.float32)
    k_tensor = trans_np_hifuint8_tensor_to_float32(k_tensor_hifp8).astype(np.float32)
    v_tensor = trans_np_hifuint8_tensor_to_float32(v_tensor_hifp8).astype(np.float32)

    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']


    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end-kvs_begin


    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("start matmul1")
        qk_i = np.matmul(q_tensor.astype(np.float32), ki.transpose(0, 1, 3, 2).astype(np.float32))

        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        print("end matmul1")


        qk_i = qk_i.astype(np.float32)
        qk_i = qk_i* np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            py_s_i = trans_np_float_tensor_to_hifuint8(in_tensor=py_s_i, round_mode="hybrid",
                                                             over_mode=True)
            py_s_i = trans_np_hifuint8_tensor_to_float32(py_s_i).astype(np.float32)
            
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)


            py_s_i = trans_np_float_tensor_to_hifuint8(in_tensor=py_s_i, round_mode="hybrid",
                                                             over_mode=True)
            py_s_i = trans_np_hifuint8_tensor_to_float32(py_s_i).astype(np.float32)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front


    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)


    return o_front,lse

def _t_pfaattention_act_innerprecise(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)



    print("start matmul1")

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]

        # mask_t = torch.from_numpy(mask_tensor)
        # mask_tensor = mask_tensor.to(bool)
        # mask_tensor = mask_tensor.cuda().requires_grad_(False)
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10 ** 38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2]+preTokens:
        ss0 = k_tensor.shape[2]+preTokens
        softmax_res[:,:,ss0:,:] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    print("start matmul2")
    bmm2Res = torch.matmul(softmax_res.to(torch.float16).to(torch.float32), v_tensor).cpu()
    print("end matmul2")
    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res
def _t_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)

    print("start matmul1")

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]

        # mask_t = torch.from_numpy(mask_tensor)
        # mask_tensor = mask_tensor.to(bool)
        # mask_tensor = mask_tensor.cuda().requires_grad_(False)
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10 ** 38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
        ss0 = k_tensor.shape[2] + preTokens
        softmax_res[:, :, ss0:, :] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    print("start matmul2")
    bmm2Res = torch.matmul(softmax_res.to(torch.bfloat16).to(torch.float32), v_tensor).cpu()
    print("end matmul2")
    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res

def _np_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param, quant_scale2=None, quant_offset2=None, antiquant_scale_k=None,
                            antiquant_scale_v=None, antiquant_offset_k=None, antiquant_offset_v=None, out_dtype="<class 'numpy.int8'>"):
    # print(
    #     f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq} preTokens:{preTokens}nextTokens:{nextTokens}")

    # qs_begin = 0
    # qs_end = q_tensor.shape[2]
    # kvs_begin = 0
    # kvs_end = k_tensor.shape[2]
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']

    qdtype = pfa_param['q_dtype']
    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    qDtype = q_tensor.dtype

    #KVcahce反量化  这块逻辑需要修改
    if k_tensor.dtype==np.int8:
        k_tensor = k_tensor.astype(np.float16)
        v_tensor = v_tensor.astype(np.float16)
        if antiquant_offset_k is not None:
            if pfa_param['anti_or_kvanti'] ==2 and pfa_param['k_antiquant_mode']==1:
                antiquant_offset_k = np.expand_dims(antiquant_offset_k[:,kvs_begin:kvs_end], axis=-1)
                antiquant_offset_v = np.expand_dims(antiquant_offset_v[:,kvs_begin:kvs_end], axis=-1)
            k_tensor += antiquant_offset_k
            v_tensor += antiquant_offset_v
        if pfa_param['anti_or_kvanti'] == 2 and pfa_param['k_antiquant_mode']==1:
            antiquant_scale_k = np.expand_dims(antiquant_scale_k[:, kvs_begin:kvs_end], axis=-1)
            antiquant_scale_v = np.expand_dims(antiquant_scale_v[:, kvs_begin:kvs_end], axis=-1)
        k_tensor *= antiquant_scale_k
        v_tensor *= antiquant_scale_v
    qkBmmRes = np.matmul(q_tensor, k_tensor.transpose([0, 1, 3, 2]))
    qkEleRes = qkBmmRes * scalar

    #这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        # print("pse_tensor",pse_tensor.shape)
        qkEleRes += pse_tensor.astype(np.float32)

    if mask_tensor is not None:
        # print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            # actul_prefix = 0
            # if 'actualprefixKV' in pfa_param:
            #     actul_prefix =  pfa_param['actualprefixKV']
            mask_tensor = mask_tensor[:, :, :(qs_end-qs_begin),:(kvs_end-kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        qkEleRes[mask_tensor.astype(np.bool_)] = -1.7 * 10 ** 38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = npSoftmax_new(qkEleRes)
    lse = None
    if 'lseflag' in pfa_param and pfa_param['lseflag']:
        lse = np.log(x_sum) + x_max

    if qdtype == np.float16:
        bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_tensor)
    elif qdtype == np.float32:
        bmm2Res = np.matmul(softmax_res, v_tensor)
    else:
        bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_tensor)

    #/sum
    bmm2Res = bmm2Res/x_sum

    #20240521新修改
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                bmm2Res[:, :, i, :] = 0
                if lse is not None:
                    lse[:,:,i,:] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        bmm2Res = bmm2Res * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float16)
        bmm2Res = np.around(bmm2Res)
        # 转成int8防止溢出
        bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(np.int8)

    # if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
    #     ss0 = k_tensor.shape[2] + preTokens
    #     bmm2Res[:, :, ss0:, :] = 0
    #
    # if nextTokens < 0:
    #     ss1 = -nextTokens
    #     bmm2Res[:, :, :ss1, :] = 0


    # print(f"return shape:{bmm2Res.shape}")
    return bmm2Res, lse


def _t_promtattention_bnsd(q_tensor, q_shape, k_tensor, k_shape, v_tensor, v_shape,  pse_tensor,mask_tensor, scale, actseqlens,
                           actkvseqlens, preTokens, nextTokens, dequant_scale1=None, dequant_scale2=None, quant_scale1=None, quant_scale2=None,
                           quant_offset2=None, out_dtype="<class 'numpy.int8'>", innerprecise=1):
    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    print(f"B:{batch_value}  actseqlens:{actseqlens_size}")
    if 0 in q_shape:
        print(f"q shape:{q_shape} is empty,skip")
        return
    y = torch.zeros(q_shape, dtype=torch.float32)
    for b_index in range(batch_value):
        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue
        # bnsd
        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                k_tensor_cur = k_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                v_tensor_cur = v_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur = mask_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if pse_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if act_seq is None:
                    if innerprecise==0:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _t_pfaattention_act_innerprecise(q_tensor_cur,k_tensor_cur,v_tensor_cur, pse_cur,
                                                                                                        mask_cur, scale,
                                                                                                        act_seq,
                                                                                                        act_kv_seq,
                                                                                                        preTokens,
                                                                                                        nextTokens,
                                                                                                        dequant_scale1,
                                                                                                        dequant_scale2,
                                                                                                        quant_scale1,
                                                                                                        quant_scale2,
                                                                                                        quant_offset2)
                    else:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _t_pfaattention_act(q_tensor_cur,
                                                                                                    k_tensor_cur,
                                                                                                    v_tensor_cur,
                                                                                                    pse_cur,
                                                                                                    mask_cur, scale,
                                                                                                    act_seq, act_kv_seq,
                                                                                                    preTokens,
                                                                                                    nextTokens,
                                                                                                    dequant_scale1,
                                                                                                    dequant_scale2,
                                                                                                    quant_scale1,
                                                                                                    quant_scale2,
                                                                                                    quant_offset2)
                else:
                    if innerprecise==0:
                        y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _t_pfaattention_act_innerprecise(
                                q_tensor_cur,k_tensor_cur,v_tensor_cur, pse_cur,mask_cur, scale,act_seq,act_kv_seq, preTokens, nextTokens,
                                dequant_scale1,dequant_scale2,quant_scale1,quant_scale2, quant_offset2)
                    else:
                        y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _t_pfaattention_act(q_tensor_cur,
                                                                                                           k_tensor_cur,
                                                                                                           v_tensor_cur,
                                                                                                           pse_cur,
                                                                                                           mask_cur,
                                                                                                           scale,
                                                                                                           act_seq,
                                                                                                           act_kv_seq,
                                                                                                           preTokens,
                                                                                                           nextTokens,
                                                                                                           dequant_scale1,
                                                                                                           dequant_scale2,
                                                                                                           quant_scale1,
                                                                                                           quant_scale2,
                                                                                                           quant_offset2)

    return y

def _np_promtattention_bnsd(q_tensor, q_shape, k_tensor_list, k_shape_list, v_tensor_list, v_shape_list, pse_bnsd_tensor, mask_tensor, scale, actseqlens,
                           actkvseqlens, preTokens_list, nextTokens_list, sparse,pfa_param, dequant_scale1=None, dequant_scale2=None, quant_scale1=None, quant_scale2=None,
                           quant_offset2=None, antiquant_scale=None, antiquant_offset=None, out_dtype="<class 'numpy.int8'>", q_dtype_origin="origin"):

    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    print(f"B:{batch_value}  actseqlens:{actseqlens_size}")
    # y = np.zeros(q_shape, dtype=np.float32)
    outShape = copy.deepcopy(q_shape)
    outShape[-1] = v_shape_list[0][-1]
    y = np.zeros(outShape, dtype=np.float32)
    lse = np.full(pfa_param['lseshape'], np.inf)
    preTokens = preTokens_list
    nextTokens = nextTokens_list
    # print(mask_tensor[0][0])
    tensor_list_flag = len(k_tensor_list)
    if mask_tensor is not None:
        mask_cur = np.zeros([1, 1, q_shape[2], len(mask_tensor[0][0])], dtype='uint8')
    if tensor_list_flag==1:
        k_tensor = k_tensor_list[0]
        k_shape = k_shape_list[0]
        v_tensor = v_tensor_list[0]
        v_shape = v_shape_list[0]
    for b_index in range(batch_value):
        if tensor_list_flag!=1:
            k_tensor = k_tensor_list[b_index]
            k_shape = k_shape_list[b_index]
            v_tensor = v_tensor_list[b_index]
            v_shape = v_shape_list[b_index]


        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue

        qs_begin = 0
        qs_end = q_tensor.shape[2]
        kvs_begin = 0
        kvs_end = k_tensor.shape[2]
        if 'queryPaddingSize' in pfa_param:
            qs_begin = int(q_tensor.shape[2] - act_seq - pfa_param['queryPaddingSize'][0])
            qs_end = int(q_tensor.shape[2] - pfa_param['queryPaddingSize'][0])
            print(f"query_left padding--- s_begin:{qs_begin}, s_end:{qs_end}")

        else:
            if act_seq is not None:
                qs_end = act_seq
        if 'kvPaddingSize' in pfa_param:
            kvs_begin = int(k_tensor.shape[2] - act_kv_seq - pfa_param['kvPaddingSize'][0])
            kvs_end = int(k_tensor.shape[2] - pfa_param['kvPaddingSize'][0])
            print(f"kv_left padding--- s_begin:{kvs_begin}, s_end:{kvs_end}")
        else:
            if act_kv_seq is not None:
                kvs_end = act_kv_seq
        pfa_param['qs_begin'] = qs_begin
        pfa_param['qs_end'] = qs_end
        pfa_param['kvs_begin'] = kvs_begin
        pfa_param['kvs_end'] = kvs_end
        if 'actualprefixKV' in pfa_param:
            if pfa_param['actualprefixKV']==0:
                pfa_param['kvs_end']+=pfa_param["shared_prefix_k"].shape[2]
            else:
                pfa_param['kvs_end']+=pfa_param['actualprefixKV']
        pfa_param['sparse'] = sparse


        # bnsd
        if sparse == 4:
            preTokens = preTokens_list[b_index]
        if sparse == 3 or sparse == 4:
            nextTokens = nextTokens_list[b_index]

        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                if tensor_list_flag==1:
                    k_tensor_cur = k_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                    v_tensor_cur = v_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                else:
                    k_tensor_cur = k_tensor[:, n_index:(n_index + 1), :, :]
                    v_tensor_cur = v_tensor[:, n_index:(n_index + 1), :, :]

                if "shared_prefix_k" in pfa_param:
                    actual_prefix_len = pfa_param['actualprefixKV']
                    if actual_prefix_len==0:
                        shared_prefix_k = pfa_param["shared_prefix_k"][:,n_index:(n_index + 1),:,:]
                        shared_prefix_v = pfa_param["shared_prefix_v"][:,n_index:(n_index + 1),:,:]
                    else:
                        shared_prefix_k = pfa_param["shared_prefix_k"][:, n_index:(n_index + 1), :actual_prefix_len, :]
                        shared_prefix_v = pfa_param["shared_prefix_v"][:, n_index:(n_index + 1), :actual_prefix_len, :]
                    k_tensor_cur = np.append(shared_prefix_k, k_tensor_cur, axis=2)
                    v_tensor_cur = np.append(shared_prefix_v, v_tensor_cur, axis=2)

                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur[0, 0, :, :] = mask_tensor[b_index]
                    # mask_cur = mask_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if pse_bnsd_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_bnsd_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                quant_scale2_cur = None
                quant_offset2_cur = None
                if quant_scale2 is not None:
                    if len(quant_scale2.shape) == 1:
                        quant_scale2_cur = quant_scale2
                    else:
                        quant_scale2_cur = quant_scale2[:, n_index:(n_index + 1), :, :]
                    if quant_offset2 is not None:
                        if len(quant_offset2.shape) == 1:
                            quant_offset2_cur = quant_offset2
                        else:
                            quant_offset2_cur = quant_offset2[:, n_index:(n_index + 1), :, :]

                antiquant_scale_k = None
                antiquant_scale_v = None
                antiquant_offset_k = None
                antiquant_offset_v = None
                if k_tensor_cur.dtype==np.int8 and q_tensor_cur.dtype!=np.int8:
                    if pfa_param['anti_or_kvanti']==1:
                        if len(antiquant_scale.shape) == 1:
                            antiquant_scale_k = antiquant_scale[0]
                            antiquant_scale_v = antiquant_scale[1]
                        else:
                            antiquant_scale_k = antiquant_scale[0,n_index:(n_index + 1),:,:]
                            antiquant_scale_v = antiquant_scale[1,n_index:(n_index + 1),:,:]
                        if antiquant_offset is not None:
                            if len(antiquant_offset.shape) ==1:
                                antiquant_offset_k = antiquant_offset[0]
                                antiquant_offset_v = antiquant_offset[1]
                            else:
                                antiquant_offset_k = antiquant_offset[0,n_index:(n_index + 1),:,:]
                                antiquant_offset_v = antiquant_offset[1,n_index:(n_index + 1),:,:]
                    if pfa_param['anti_or_kvanti']==2:#分离量化参数
                        if pfa_param['k_antiquant_mode'] ==0 :
                            if len(pfa_param['k_antiquant_scale'].shape) == 1:
                                antiquant_scale_k = pfa_param['k_antiquant_scale']
                                antiquant_scale_v = pfa_param['v_antiquant_scale']
                            else:
                                antiquant_scale_k = pfa_param['k_antiquant_scale'][n_index:(n_index + 1),:,:]
                                antiquant_scale_v = pfa_param['v_antiquant_scale'][n_index:(n_index + 1),:,:]
                            if pfa_param['k_antiquant_offset'] is not None:
                                if len(pfa_param['k_antiquant_offset'].shape) ==1:#petensor
                                    antiquant_offset_k = pfa_param['k_antiquant_offset']
                                    antiquant_offset_v = pfa_param['v_antiquant_offset']
                                else:
                                    antiquant_offset_k = pfa_param['k_antiquant_offset'][n_index:(n_index + 1),:,:]
                                    antiquant_offset_v = pfa_param['v_antiquant_offset'][n_index:(n_index + 1),:,:]
                        if pfa_param['k_antiquant_mode'] == 1:
                            antiquant_scale_k = pfa_param['k_antiquant_scale'][b_index:(b_index + 1),  :]
                            antiquant_scale_v = pfa_param['v_antiquant_scale'][b_index:(b_index + 1),  :]
                            if pfa_param['k_antiquant_offset'] is not None:
                                antiquant_offset_k = pfa_param['k_antiquant_offset'][b_index:(b_index + 1), :]
                                antiquant_offset_v = pfa_param['v_antiquant_offset'][b_index:(b_index + 1), :]



                if q_tensor.dtype == "int8":
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_int8(q_tensor_cur,
                                                                                                      k_tensor_cur,
                                                                                                      v_tensor_cur,
                                                                                                      pse_cur,
                                                                                                      mask_cur,
                                                                                                      scale,
                                                                                                      act_seq,
                                                                                                      act_kv_seq,
                                                                                                      preTokens,
                                                                                                      nextTokens,
                                                                                                      pfa_param,
                                                                                                      dequant_scale1,
                                                                                                      dequant_scale2,
                                                                                                      quant_scale1,
                                                                                                      quant_scale2_cur,
                                                                                                      quant_offset2_cur,
                                                                                                      out_dtype)
                elif q_tensor.dtype == "float8_e5m2" or q_tensor.dtype == "float8_e4m3fn":
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_fp8(q_tensor_cur,
                                                                                                      k_tensor_cur,
                                                                                                      v_tensor_cur,
                                                                                                      pse_cur,
                                                                                                      mask_cur,
                                                                                                      scale,
                                                                                                      act_seq,
                                                                                                      act_kv_seq,
                                                                                                      preTokens,
                                                                                                      nextTokens,
                                                                                                      pfa_param,
                                                                                                      dequant_scale1,
                                                                                                      dequant_scale2,
                                                                                                      quant_scale1,
                                                                                                      quant_scale2_cur,
                                                                                                      quant_offset2_cur,
                                                                                                      out_dtype)
                elif q_dtype_origin == "hifloat8":
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_hifp8(q_tensor_cur,
                                                                                                      k_tensor_cur,
                                                                                                      v_tensor_cur,
                                                                                                      pse_cur,
                                                                                                      mask_cur,
                                                                                                      scale,
                                                                                                      act_seq,
                                                                                                      act_kv_seq,
                                                                                                      preTokens,
                                                                                                      nextTokens,
                                                                                                      pfa_param,
                                                                                                      dequant_scale1,
                                                                                                      dequant_scale2,
                                                                                                      quant_scale1,
                                                                                                      quant_scale2_cur,
                                                                                                      quant_offset2_cur,
                                                                                                      out_dtype)
                else:
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :], lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act(q_tensor_cur,
                                                                                                 k_tensor_cur,
                                                                                                 v_tensor_cur,
                                                                                                 pse_cur,
                                                                                                 mask_cur, scale,
                                                                                                 act_seq,
                                                                                                 act_kv_seq,
                                                                                                 preTokens,
                                                                                                 nextTokens,
                                                                                                 pfa_param,
                                                                                                 quant_scale2_cur,
                                                                                                 quant_offset2_cur,
                                                                                                 antiquant_scale_k,
                                                                                                 antiquant_scale_v,
                                                                                                 antiquant_offset_k,
                                                                                                 antiquant_offset_v,
                                                                                                 out_dtype)



    return y,lse

def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value

    result = (data_hex // 8192) * 8192

    return result


def trans_19bit(deqscale):
    res_19bit = np.zeros(deqscale.shape[0], dtype=np.uint64)

    for idx, scale in enumerate(deqscale):
        val = np.bitwise_and(((conv_float_to_u32(scale) >> 13) << 13), 0xffffffff)  # [31:13], M
        val = val.astype(np.uint64)
        res_19bit[idx] = val

    return res_19bit

def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else: #BNSD
        return q_shape[2]

def aclnnPromptFlashAttention_NPU_unification(torch_tensor_list, params):
    import torch_npu
    q_tensor = torch_tensor_list[0].npu()
    k_tensor = torch_tensor_list[1].npu()
    v_tensor = torch_tensor_list[2].npu()
    q_rope = None
    k_rope = None
    if torch_tensor_list[3] is not None:
        q_rope = torch_tensor_list[3].npu()
        k_rope = torch_tensor_list[4].npu()

    m_tensor = None

    num_heads = None
    input_layout = None
    scale_value = None
    actual_seq_lengths = None
    num_key_value_heads = None
    pre_tokens = None
    next_tokens = None
    actual_seq_lengths_kv = None
    sparse_mode = None

    num_heads = params['numheads']
    flagList = params['flaglist']
    if flagList[3]:
        p_tensor = torch_tensor_list[3]
    if flagList[4]:
        case_name = params["case_name"]
        m_tensor = torch.load('{}_attenmask.pt'.format(case_name)).npu()

    if flagList[5]:
        actual_seq_lengths = params["actualseqlengths"]
    if flagList[6]:
        actual_seq_lengths_kv = params["actualseqlengthskv"]
    # if flagList[13]:
    #     scale_value = params["scalevalue"]
    # if flagList[14]:
    #     pre_tokens = params["pretokens"]
    # if flagList[15]:
    #     next_tokens = params["nexttokens"]
    # if flagList[16]:
    #     input_layout = params["inputlayout"]
    # if flagList[16]:
    #     num_key_value_heads = params["numkeyvalueheads"]
    # if flagList[17]:
    #     sparse_mode = params['sparsemode']

    scale_value = params['scalevalue'] if 'scalevalue' in params else 0
    pre_tokens = params['pretokens'] if 'pretokens' in params else 0
    next_tokens = params['nexttokens'] if 'nexttokens' in params else 0
    input_layout = params['inputlayout'] if 'inputlayout' in params else 0
    num_key_value_heads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
    sparse_mode = params['sparsemode'] if 'sparsemode' in params else 0

    # torch_npu.npu_prompt_flash_attention
    # torch.ops.npu.npu_prompt_flash_attention
    # sparse_mode = 10
    npu_out = torch_npu.npu_prompt_flash_attention(
            q_tensor, k_tensor, v_tensor,
            num_heads = num_heads, scale_value=scale_value, atten_mask = m_tensor, sparse_mode = 10, input_layout = input_layout, pre_tokens=2147483647, next_tokens=0,num_key_value_heads=num_key_value_heads, actual_seq_lengths = actual_seq_lengths, actual_seq_lengths_kv = actual_seq_lengths_kv)
    # npu_out, _ = torch_npu.npu_fused_infer_attention_score_v2(q_tensor, k_tensor, v_tensor, atten_mask=m_tensor, query_rope=q_rope ,key_rope=k_rope,
    #                                                             actual_seq_qlen = actual_seq_lengths, actual_seq_kvlen = actual_seq_lengths_kv,
    #                                                             num_query_heads = num_heads, num_key_value_heads = num_key_value_heads,input_layout = input_layout, softmax_scale = scale_value, pre_tokens=2147483647, next_tokens=0, inner_precise=0)
    # npu_out = torch_npu.npu_prompt_flash_attention(q_tensor, k_tensor, v_tensor, atten_mask=m_tensor,
    #                                                actual_seq_lengths=actual_seq_lengths, num_heads=num_heads,
    #                                                scale_value=scale_value, pre_tokens=pre_tokens,
    #                                                next_tokens=next_tokens, input_layout=input_layout,
    #                                                num_key_value_heads=num_key_value_heads,
    #                                                actual_seq_lengths_kv=actual_seq_lengths_kv,
    #                                                sparse_mode=sparse_mode).cpu()
    return npu_out

def kvantiquantmode(flagList,torch_tensor_list,params,v_end_index,numKeyValueHeads):
    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    anti_or_kvanti = 0
    # if len(flagList) > 12:
    # 获取kv量化参数
    flagtensor = 0
    if flagList[12]:
        anti_or_kvanti = 1
        antiquantscale_shape = params['shape_input'][v_end_index + 8]
        antiquantscale_tensor = torch_tensor_list[v_end_index + 8]
        if len(antiquantscale_shape) == 1:  # pertensor
            print("antiquantscale_pertensor")
            antiquant_scale = antiquantscale_tensor
        else:  # perchanel
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_2n1d_tensor = _t_trans_2h_to_2n1d(antiquantscale_shape, antiquantscale_tensor,
                                                           numKeyValueHeads)

            antiquant_scale = antiquantscale_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[13]:
        antiquantoffset_shape = params['shape_input'][v_end_index + 9]
        antiquantoffset_tensor = torch_tensor_list[v_end_index + 9]
        if len(antiquantoffset_shape) == 1:
            print("antiquantoffset_pertensor")
            antiquant_offset = antiquantoffset_tensor
        else:
            antiquantoffset_2n1d_tensor = _t_trans_2h_to_2n1d(antiquantoffset_shape, antiquantoffset_tensor,
                                                            numKeyValueHeads)

            antiquant_offset = antiquantoffset_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[12]:
        return antiquant_scale[:1],antiquant_scale[1:2],antiquant_offset[:1],antiquant_offset[1:2]
    # 判断kv分离的伪量化
    if flagList[17] or flagList[19]:
        anti_or_kvanti = 2
        q_dtype_np = np.float16
        k_antiquantscale_shape = params['shape_input'][v_end_index + 13]
        k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
        v_antiquantscale_shape = params['shape_input'][v_end_index + 15]
        v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]

        if params['k_antiquant_mode'] == 0:
            if len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1:  # pertensor
                print("antiquantscale_pertensor")
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
            else:  # perchanel
                # 将kv量化参数转换为n1d(匹配bnsd)
                k_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,
                                                              numKeyValueHeads)
                v_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,
                                                              numKeyValueHeads)


                k_antiquant_scale = k_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_scale = v_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
        if params['k_antiquant_mode'] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_scale = k_antiquantscale_tensor.reshape(k_antiquantscale_tensor.shape[0],k_antiquantscale_tensor.shape[1],1,1)
            v_antiquant_scale = v_antiquantscale_tensor.reshape(v_antiquantscale_tensor.shape[0],v_antiquantscale_tensor.shape[1],1,1)

    if flagList[18] or flagList[20]:
        q_dtype_np = np.float16
        k_antiquantoffset_shape = params['shape_input'][v_end_index + 14]
        k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
        v_antiquantoffset_shape = params['shape_input'][v_end_index + 16]
        v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

        if params['k_antiquant_mode'] == 0:
            if len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1:  # pertensor
                print("antiquantscale_pertensor")
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
            else:  # perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                k_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(k_antiquantoffset_shape,
                                                               k_antiquantoffset_tensor, numKeyValueHeads)
                v_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(v_antiquantoffset_shape,
                                                               v_antiquantoffset_tensor, numKeyValueHeads)


                k_antiquant_offset = k_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_offset = v_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
        if params['k_antiquant_mode'] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_offset = k_antiquantoffset_tensor.reshape(k_antiquantoffset_tensor.shape[0],k_antiquantoffset_tensor.shape[1],1,1)
            v_antiquant_offset = v_antiquantoffset_tensor.reshape(v_antiquantoffset_tensor.shape[0],v_antiquantoffset_tensor.shape[1],1,1)
    if flagList[17]:
        return k_antiquant_scale,v_antiquant_scale,k_antiquant_offset,v_antiquant_offset
    return k_antiquant_scale,v_antiquant_scale,k_antiquant_offset,v_antiquant_offset
def aclnnPromptFlashAttention_GPU_unification(torch_tensor_list, params):
    enable_gpu = params['enablegpu']
    print("#" * 20, "enable_gpu")
    print(enable_gpu)
    innerprecise = params['innerprecise']
    if enable_gpu == "True" or enable_gpu == "TRUE":
        print("#" * 20)
        print("gpu biggggggg")
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache
        from flash_attn.flash_attn_interface import flash_attn_func
        # 设置默认值
        # scaleValue = None
        # numKeyValueHeads = 0
        # sp_mode = 0
        flagList = params['flaglist']

        scaleValue = params['scalevalue'] if 'scalevalue' in params else 0
        preTokens = params['pretokens'] if 'pretokens' in params else 0
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 0
        inputLayout = params['inputlayout'] if 'inputlayout' in params else 0
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0


        q_shape = params["shape_input"][0]
        k_shape = params["shape_input"][1]
        # inputLayout = params["inputlayout"]
        numHeads = params["numheads"]

        actualSeqLengthsKV = []
        actualSeqLengths = params['actualseqlengths']
        if numKeyValueHeads == 0:
            numKeyValueHeads = numHeads
        if inputLayout == "SH":
            actualSeqLengthsKV = actualSeqLengths
        else:
            actualSeqLengthsKV = params['actualseqlengthskv']

        if inputLayout == "TND":
            # 将tnd格式下的act seq转成普通的act seq
            actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
            actualSeqLengthsKV = trans_tnd_actseq(actualSeqLengthsKV)

        # k和v的shape_list只能通过抛去已知tensor个数，除2计算
        num_input = len(params['shape_input'])
        tensornum = 23
        # if len(flagList) > 23:
        #     tensornum = 14
        # if len(flagList) > 25:
        #     tensornum = 16
        # if len(flagList) > 28:
        #     tensornum = 20
        k_shape_num = v_shape_num = int((num_input - tensornum) / 2)
        print(f"k_shape_num:{k_shape_num}")

        k_start_index = 1
        k_end_index = int(k_shape_num)
        v_start_index = int(k_shape_num) + 1
        v_end_index = int(k_shape_num + v_shape_num)


        # 转成gpu需要的bsnd
        q_tensor, q_bnsd_shape = t_bsh_to_bsnd(torch_tensor_list[0], q_shape, numHeads, actualSeqLengths, inputLayout)

        k_tensor, k_bnsd_shape = t_bsh_to_bsnd(torch_tensor_list[1], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                               inputLayout)
        v_tensor, v_bnsd_shape = t_bsh_to_bsnd(torch_tensor_list[2], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                               inputLayout)

        # if len(flagList) > 23:
        if flagList[15]:#queryPaddingSize
            pass
            #fanzhuanq
        if flagList[16]:#kvPaddingSize
            pass
            #翻转kv
            # pfa_param['kvPaddingSize'] = torch_tensor_list[v_end_index + 14]

        #判断是否伪量化
        k_antiquant_scale,v_antiquant_scale,k_antiquant_offset,v_antiquant_offset = kvantiquantmode(flagList,torch_tensor_list,params,v_end_index,numKeyValueHeads)

        # KVcahce反量化  这块逻辑需要修改
        if q_tensor.dtype != torch.int8 and k_tensor.dtype == torch.int8:
            k_tensor = k_tensor.to(torch.float16)
            v_tensor = v_tensor.to(torch.float16)
            if k_antiquant_offset is not None:
                k_tensor += k_antiquant_offset
                v_tensor += v_antiquant_offset
            k_tensor *= k_antiquant_scale
            v_tensor *= v_antiquant_scale
            k_tensor = k_tensor.to(torch.bfloat16)
            v_tensor = v_tensor.to(torch.bfloat16)
        pse = None
        if flagList[3]:
            pse = get_slopes(numHeads).cuda().requires_grad_(False)
        k_cache = k_tensor
        v_cache = v_tensor
        q_tensor = q_tensor.cuda().requires_grad_(False)
        k_cache = k_cache.cuda().requires_grad_(False)
        v_cache = v_cache.cuda().requires_grad_(False)

        if len(actualSeqLengthsKV) == 0:
            actualSeqLengthsKV = None
        else:
            actualSeqLengthsKV = torch.from_numpy(np.array(actualSeqLengthsKV)).to(torch.int32)
            actualSeqLengthsKV = actualSeqLengthsKV.cuda().requires_grad_(False)

        causal_switch = False
        window_size0 = -1
        window_size1 = -1
        qs = q_bnsd_shape[1]
        kvs = k_bnsd_shape[1]
        if sp_mode == 0:
            if actualSeqLengthsKV is not None:
                window_size0 = preTokens + actualSeqLengthsKV[0] - qs
                window_size1 = nextTokens - actualSeqLengthsKV[0] + qs
            else:
                window_size0 = preTokens + kvs - qs  # 65535+4096-512
                window_size1 = nextTokens - kvs + qs  # 0-4096+512
        if sp_mode == 2:
            if actualSeqLengthsKV is not None:
                window_size1 = 0 - actualSeqLengthsKV[0] + qs
            else:
                window_size1 = 0 - kvs + qs
        if sp_mode == 3:
            causal_switch = True
        if sp_mode == 4:
            window_size0 = preTokens
            window_size1 = nextTokens

        if ("softmaxlseflag" in params and params["softmaxlseflag"] == True) or (
                "softmax_lse_flag" in params and params["softmax_lse_flag"] == True):
            out, lse, smask = flash_attn_func(q=q_tensor, k=k_cache, v=v_cache,
                                              softmax_scale=scaleValue, causal=causal_switch,
                                              window_size=(window_size0, window_size1), alibi_slopes=pse,
                                              return_attn_probs=True)
            lse = lse.cpu()
        else:
            out = flash_attn_with_kvcache(q=q_tensor, k_cache=k_cache, v_cache=v_cache, k=None, v=None,
                                          cache_seqlens=actualSeqLengthsKV,
                                          softmax_scale=scaleValue, causal=causal_switch, alibi_slopes=pse,
                                          window_size=(window_size0, window_size1), num_splits=0)

        y_all = out.cpu()

        # 增加quant
        if flagList[10]:
            quant_scale2 = torch_tensor_list[6]
            quant_scale2_shape = params['shape_input'][6]
            per_channel = True
            if len(quant_scale2_shape) == 1:
                if quant_scale2_shape[0] == 1:
                    per_channel = False
            if per_channel:
                quant_scale2 = _t_trans_1h_to_1n1d(quant_scale2_shape, quant_scale2,
                                                   numHeads, inputLayout)
            y_all = y_all * quant_scale2.cpu().to(torch.float16)
        if flagList[11]:
            quant_offset2 = torch_tensor_list[7]
            quant_offset2_shape = params['shape_input'][7]
            per_channel = True
            if len(quant_offset2_shape) == 1:
                if quant_offset2_shape[0] == 1:
                    per_channel = False
            if per_channel:
                quant_offset2 = _t_trans_1h_to_1n1d(quant_offset2_shape, quant_offset2,
                                                    numHeads, inputLayout)
            y_all += quant_offset2.to(torch.float16)
        if flagList[10]:
            y_all = torch.round(y_all)
            # 转成int8防止溢出
            y_all = torch.where(y_all > 127, 127, y_all)
            y_all = torch.where(y_all < -128, -128, y_all)
            y_all = y_all.to(torch.int8)

        out_shape = q_shape
        if inputLayout == "BSH":
            y_all = y_all.reshape(out_shape)
        elif inputLayout == "NSD":
            y_all = y_all.permute(0, 2, 1, 3).reshape(out_shape)
        elif inputLayout == "TND":
            T = sum(actualSeqLengths)
            B = len(actualSeqLengths)
            N = out_shape[1]
            D = out_shape[2]
            output = np.zeros((T, N, D), dtype=y_all.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
                t_start += act_s
            y_all = output
        elif len(out_shape) == 2:
            if q_bnsd_shape[0] == 1:
                y_all = y_all.reshape(out_shape).astype(np.float32)
            else:
                # print(y_all)
                sums = 0
                B = q_bnsd_shape[0]
                N = q_bnsd_shape[1]
                S = q_bnsd_shape[2]
                D = q_bnsd_shape[3]
                # y_all = y_all.transpose(0, 2, 1, 3)
                yNewAll = torch.zeros((S, N * D))
                for i in range(B):
                    for j in range(actualSeqLengths[i]):
                        yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                    sums += actualSeqLengths[i]
                y_all = yNewAll.astype(np.float32)
        elif inputLayout == "BNSD":
            y_all = y_all.permute([0, 2, 1, 3])
        else:
            y_all = y_all
        if ("softmaxlseflag" in params and params["softmaxlseflag"] == True) or (
                "softmax_lse_flag" in params and params["softmax_lse_flag"] == True):
            return y_all, lse
        else:
            return y_all
    elif enable_gpu == "SMALL" or enable_gpu == "False" or enable_gpu == "FALSE":  # 走小算子
        print("#" * 20)
        print("small")
        flagList = params['flaglist']
        actualSeqLengths = params['actualseqlengths']
        actualSeqLengthsKV = None
        numHeads = params['numheads']
        # 设置参数默认值
        scaleValue = 1
        preTokens = 214748647
        nextTokens = 0
        inputLayout = "BSH"
        numKeyValueHeads = 0
        sp_mode = 0
        # if flagList[13]:
        #     scaleValue = params['scalevalue']
        # if flagList[14]:
        #     preTokens = params['pretokens']
        # if flagList[15]:
        #     nextTokens = params['nexttokens']
        # if flagList[16]:
        #     inputLayout = params['inputlayout']
        # if flagList[17]:
        #     numKeyValueHeads = params['numkeyvalueheads']
        # if flagList[18]:
        #     sp_mode = params['sparsemode']
        scaleValue = params['scalevalue'] if 'scalevalue' in params else 0
        preTokens = params['pretokens'] if 'pretokens' in params else 0
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 0
        inputLayout = params['inputlayout'] if 'inputlayout' in params else 0
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0

        if numKeyValueHeads == 0:
            numKeyValueHeads = numHeads

        q_shape = params['shape_input'][0]
        q_dtype = np.float16  # numpy类型
        out_dtype = np.float16  # numpy类型

        if inputLayout == "SH":
            actualSeqLengthsKV = actualSeqLengths
        else:
            actualSeqLengthsKV = params['actualseqlengthskv']
        # q_tensor,q_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[0],q_shape,numHeads)
        q_tensor, q_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[0], q_shape, numHeads, actualSeqLengths,
                                                      inputLayout)
        kv_empty_flag = False
        k_shape = params['shape_input'][1]

        if 0 in k_shape or len(k_shape) == 0:
            kv_empty_flag = True
        else:
            # k_tensor,k_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[1],k_shape,numKeyValueHeads)
            k_tensor, k_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[1], k_shape, numKeyValueHeads,
                                                          actualSeqLengthsKV,
                                                          inputLayout)
            if numKeyValueHeads != numHeads:
                print("broadcast k")
                k_dtype = np.float16
                print(k_dtype)
                k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor)
                k_tensor = k_tensor.to(k_dtype)
        v_shape = params['shape_input'][2]
        if 0 in v_shape or len(v_shape) == 0 or kv_empty_flag:
            kv_empty_flag = True
        else:
            # v_tensor,v_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[2],k_shape,numKeyValueHeads)
            v_tensor, v_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[2], k_shape, numKeyValueHeads,
                                                          actualSeqLengthsKV,
                                                          inputLayout)
            if numKeyValueHeads != numHeads:
                print("broadcast v")
                v_dtype = np.float16
                v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor)
                v_tensor = v_tensor.to(k_dtype)
        pse_bnsd_tensor = None
        pse_shift_shape = params['shape_input'][3]
        if flagList[3] == 0 or 0 in pse_shift_shape:
            pse_bnsd_tensor = None
        else:
            pse_shift = torch_tensor_list[3]

            pse_bnsd_tensor = _t_broadcast_pseShift_n(pse_shift, pse_shift_shape, q_bnsd_shape[0])  # to bnsd

        m_bnsd_tensor = None
        npu_m_shape = params['shape_input'][4]
        qs = q_bnsd_shape[2]
        kvs = k_bnsd_shape[2]
        m_dtype = np.float16
        if flagList[4] == 0 or 0 in npu_m_shape:
            m_bnsd_tensor = None
        else:
            batch = q_bnsd_shape[0]
            numheads = q_bnsd_shape[1]
            if sp_mode == 0 or sp_mode == 1:
                cpu_m_shape = npu_m_shape
            else:  # sparse 为leftUpCausal2  rightDownCausal3  band4
                cpu_m_shape = [qs, kvs]  # cpu
                if sp_mode != 4:
                    preTokens = 214748647

            cpu_m_tensor = _t_gpu_create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, preTokens,
                                                              nextTokens,
                                                              actualSeqLengths, actualSeqLengthsKV, batch,
                                                              numheads, sp_mode, random_ones=0)
            if sp_mode != 3:
                m_bnsd_tensor = _t_broadcast_mask_n(cpu_m_tensor, cpu_m_shape, numHeads, q_bnsd_shape[0])
            else:
                m_bnsd_tensor = cpu_m_tensor

        dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = None, None, None, None, None

        out_shape = q_shape
        if kv_empty_flag:
            return torch.zeros(out_shape)
        else:
            y_all = _t_promtattention_bnsd(q_tensor, q_bnsd_shape, k_tensor, k_bnsd_shape, v_tensor, v_bnsd_shape,
                                           pse_bnsd_tensor,
                                           m_bnsd_tensor, scaleValue, actualSeqLengths, actualSeqLengthsKV, preTokens,
                                           nextTokens,
                                           dequant_scale1,
                                           dequant_scale2, quant_scale1, quant_scale2, quant_offset2, out_dtype,
                                           innerprecise)
            if inputLayout == "BSH":
                y_all = y_all.permute(0, 2, 1, 3).reshape(out_shape)
            elif inputLayout == "NSD":
                y_all = y_all.reshape(out_shape)
            elif inputLayout == "TND":
                T = sum(actualSeqLengths)
                B = len(actualSeqLengths)
                N = out_shape[1]
                D = out_shape[2]
                output = np.zeros((T, N, D), dtype=y_all.dtype)
                t_start = 0
                for b_index in range(B):
                    act_s = actualSeqLengths[b_index]
                    t_end = t_start + act_s
                    if act_s == 0:
                        continue
                    for n_index in range(N):
                        output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
                    t_start += act_s
                y_all = output
            elif len(out_shape) == 2:
                if q_bnsd_shape[0] == 1:
                    y_all = y_all.permute(0, 2, 1, 3).reshape(out_shape).to(torch.float32)
                else:
                    # print(y_all)
                    sums = 0
                    B = q_bnsd_shape[0]
                    N = q_bnsd_shape[1]
                    S = q_bnsd_shape[2]
                    D = q_bnsd_shape[3]
                    y_all = y_all.permute(0, 2, 1, 3)
                    yNewAll = torch.zeros((S, N * D))
                    for i in range(B):
                        for j in range(actualSeqLengths[i]):
                            yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                        sums += actualSeqLengths[i]
                    y_all = yNewAll.to(torch.float32)
            else:
                y_all = y_all
            # y_all = torch.from_numpy(y_all)
            print(y_all)
            return y_all
def get_attention_mask_batch_num(npu_m_shape,q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape)==2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead,s1,s2
    if len(npu_m_shape)==3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead,s1,s2
    if len(npu_m_shape)==4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead,s1,s2


def _trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:#2N1D
        print("2N1D")
        return tensor
    elif len(shape) == 2:#2H
        d_num = shape[1] // numKeyValueHeads
        print(f"[INFO]_n_trans_2h_to_2n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:#2ND
        print("2ND")
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2,1,N,D).transpose(0, 2, 1, 3)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_2h_to_2n1d : len(shape):{len(shape)}")
        return tensor

def _t_trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:#2N1D
        print("2N1D")
        return tensor
    elif len(shape) == 2:#2H
        d_num = shape[1] // numKeyValueHeads
        print(f"[INFO]_n_trans_2h_to_2n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).permute(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:#2ND
        print("2ND")
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2,1,N,D).permute(0, 2, 1, 3)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_2h_to_2n1d : len(shape):{len(shape)}")
        return tensor

def _trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:#N1D
        print("N1D")
        return tensor
    elif len(shape) == 1:#H
        d_num = shape[0] // numKeyValueHeads
        print(f"[INFO]_n_trans_h_to_n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).transpose(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:#ND
        print("ND")
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1,N,D).transpose(1, 0, 2)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_1h_to_1n1d : len(shape):{len(shape)}")
        return tensor
def _t_trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:#N1D
        print("N1D")
        return tensor
    elif len(shape) == 1:#H
        d_num = shape[0] // numKeyValueHeads
        print(f"[INFO]_n_trans_h_to_n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).permute(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:#ND
        print("ND")
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1,N,D).permute(1, 0, 2)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_1h_to_1n1d : len(shape):{len(shape)}")
        return tensor

def _t_trans_1h_to_1n1d(shape, tensor, numHeads,inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout=="BSND": #1,1,n,d
            return tensor
        else:   #BNSD 1N1D
            return tensor.transpose(1,2)
    elif len(shape) == 3:
        if inputLayout =="BSND": # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1,1,n,d)
        elif inputLayout == "BNSD" or inputLayout == "NSD": #n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1,n,1,d).transpose(1,2)
        else: #bsh 11h->1n1d
            h = shape[2]
            d = h//numHeads
            return  tensor.reshape(1,1,numHeads,d)
    elif len(shape) == 2:
        if inputLayout=="BSH":#1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d)
        else: #BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h//numHeads
        return tensor.reshape(1,1,numHeads,d)
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        exit(1)

def _trans_1h_to_1n1d(shape, tensor, numHeads,inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout=="BSND": #1,1,n,d
            return tensor.transpose(0,2,1,3)
        else:   #BNSD 1N1D
            return tensor
    elif len(shape) == 3:
        if inputLayout =="BSND": # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1,1,n,d).transpose(0,2,1,3)
        elif inputLayout == "BNSD" or inputLayout == "NSD" or inputLayout == "BNSD_BSND": #n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1,n,1,d)
        else: #bsh 11h->1n1d
            h = shape[2]
            d = h//numHeads
            return  tensor.reshape(1,1,numHeads,d).transpose(0,2,1,3)
    elif len(shape) == 2:
        if inputLayout=="BSH":#1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d).transpose(0,2,1,3)
        else: #BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d).transpose(0,2,1,3)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h//numHeads
        return tensor.reshape(1,1,numHeads,d).transpose(0,2,1,3)
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        exit(1)

def gen_outshape(layout,qshape,vshape,numKeyValueHeads,numHeads):
    outshape = copy.deepcopy(qshape)
    if layout=="BSH":
        vd = int(vshape[-1]/numKeyValueHeads)
        outshape[-1] = vd*numHeads
        return outshape
    else:
        outshape[-1] = vshape[-1]
        return outshape

def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}")
        return None, None
    elif len(shape1) == 2:
        # 量化参数1h
        if shape1[0] != shape2[0]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = tensor1.reshape(shape1[0], n, d1)
        tensor2_1nd = tensor2.reshape(shape2[0], n, d2)
        concatenated_tensor = np.concatenate((tensor1_1nd, tensor2_1nd), axis=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        if tnd_flag:
            concatenated_tensor = np.concatenate((tensor1, tensor2), axis=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = np.concatenate((tensor1_bsnd, tensor2_bsnd), axis=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        concatenated_tensor = np.concatenate((tensor1, tensor2), axis=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape

def deepseek_preprocessing(params, pfa_param, torch_tensor_list, numHeads,numKeyValueHeads,tnd_flag):
    v_end_index = pfa_param['v_end_index']
    # >> q rope info
    q_rope_tensor_shape= params['shape_input'][v_end_index + 21]
    q_rope_dtype = params['dtype_input'][v_end_index + 21]
    q_rope_tensor = torch_tensor_list[v_end_index + 21]

    # >> k rope info
    k_rope_shape = params['shape_input'][v_end_index + 22]
    k_rope_dtype = params['dtype_input'][v_end_index + 22]
    k_rope_tensor = torch_tensor_list[v_end_index + 22]

    # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    # if not ifa_param['in_quant_flag']:
    q_new_tensor, q_new_shape = concat_tensor(torch_tensor_list[0], params['shape_input'][0],
                                              q_rope_tensor,
                                              q_rope_tensor_shape, numHeads, tnd_flag)
    k_new_tensor, k_new_shape = concat_tensor(torch_tensor_list[1], params['shape_input'][1],
                                              k_rope_tensor,
                                              k_rope_shape, numKeyValueHeads, tnd_flag)

    pfa_param['q_tensor'] = q_new_tensor
    pfa_param['q_shape'] = q_new_shape
    pfa_param['k_tensor_list'][0] = k_new_tensor
    pfa_param['k_shape_list'][0] = k_new_shape

    return pfa_param

def get_param_fus(torch_tensor_list, params):
    pfa_param = {}
    # ===参数获取===
    flag_list = params['flaglist']
    inputLayout = params['inputlayout']
    actualSeqLengths = params['actualseqlengths']
    q_shape = params['shape_input'][0]

    tnd_flag = False
    if inputLayout in ["TND", "TND_NTD"]:
        tnd_flag = True
        batch = len(actualSeqLengths)
    else:
        batch = q_shape[0]
    pfa_param['batch'] = batch

    pfa_param['q_tensor'] = torch_tensor_list[0]
    pfa_param['q_shape'] = params['shape_input'][0]

    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params['shape_input'][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 == 1:
        k_shape_num = v_shape_num = batch
    else:
        k_shape_num = v_shape_num = 1
    pfa_param['v_end_index'] = v_end_index = int(k_shape_num + v_shape_num)
    pfa_param['tnd_flag'] = tnd_flag
    pfa_param['k_num'] = k_shape_num
    pfa_param['k_start_index'] = k_start_index = 1
    pfa_param['k_end_index'] = k_end_index = int(k_shape_num)
    pfa_param['v_start_index'] = v_start_index = int(k_shape_num) + 1
    print(f"k_start_index:{k_start_index} k_end_index:{k_end_index} v_start_index:{v_start_index} v_end_index:{v_end_index}")

    pfa_param['k_shape_list'] = k_ori_shape_list = params['shape_input'][k_start_index:k_end_index + 1]
    pfa_param['v_shape_list'] = v_ori_shape_list = params['shape_input'][v_start_index:v_end_index + 1]
    pfa_param['k_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]
    pfa_param['v_tensor_list'] = torch_tensor_list[v_start_index:v_end_index + 1]


    return pfa_param
def aclnnPromptFlashAttention_unification(torch_tensor_list, params):
    # torch_tensor_list ---tensor  params--attr参数
    pfa_param = {}

    action_type = params["action_type"]
    gold = ["bm_output", "bm_output_gold", "bm_gold"]

    flagList = params['flaglist']
    actualSeqLengths = params['actualseqlengths']
    actualSeqLengthsKV = params['actualseqlengthskv']
    numHeads = int(params['numheads'])
    # 设置参数默认值

    if flagList[25] == 0:
        inputLayout = 'BSH'
        #所有的参数的默认值都给写在这里
        numKeyValueHeads = 0
        scaleValue = 1.0
        preTokens = 214748647
        nextTokens = 214748647
        sp_mode = 0
        block_size =0
    else:
        scaleValue = params['scalevalue'] if 'scalevalue' in params else 1
        preTokens = params['pretokens'] if 'pretokens' in params else 214748647
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 214748647
        inputLayout = params['inputlayout'] if 'inputlayout' in params else "BSH"
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0

    if numKeyValueHeads == 0:
        numKeyValueHeads = numHeads

    pfa_param = get_param_fus(torch_tensor_list, params)
    q_shape = pfa_param['q_shape']
    q_dtype = np.float16  # numpy类型
    pfa_param['q_dtype'] = q_dtype
    batch = pfa_param["batch"]
    tnd_flag = pfa_param["tnd_flag"]

    kv_inputlayout = inputLayout
    if inputLayout=="TND" and flagList[14]:
        kv_inputlayout = "BNSD"

    if inputLayout == "SH":
        actualSeqLengthsKV = actualSeqLengths
    else:
        actualSeqLengthsKV = params['actualseqlengthskv']
    if inputLayout == "TND":
        # 将tnd格式下的act seq转成普通的act seq
        actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
    if inputLayout == "TND" and  flagList[14] == 0: #TND非PA场景需要转成普通actseq
        actualSeqLengthsKV = trans_tnd_actseq(actualSeqLengthsKV)
    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    if inputLayout != "TND":
        if actualSeqLengths != None:
            if len(actualSeqLengths) == 1 and len(actualSeqLengths) != q_shape[0]:
                actualSeqLengths_item = actualSeqLengths[0]
                for b_count in range(batch - 1):
                    actualSeqLengths.append(actualSeqLengths_item)
            # >> actualSeqLengths预处理：actualSeqLengths长度超过
            if len(actualSeqLengths) > batch:
                actualSeqLengths = actualSeqLengths[:batch]
        if actualSeqLengthsKV != None:
            if len(actualSeqLengthsKV) == 1 and len(actualSeqLengthsKV) != q_shape[0]:
                actualSeqLengthsKV_item = actualSeqLengthsKV[0]
                for b_count in range(batch - 1):
                    actualSeqLengthsKV.append(actualSeqLengthsKV_item)
            # >> actualSeqLengths预处理：actualSeqLengths长度超过
            if len(actualSeqLengthsKV) > batch:
                actualSeqLengthsKV = actualSeqLengthsKV[:batch]

    k_shape_num = pfa_param['k_num']
    v_shape = params['shape_input'][1 + k_shape_num]
    out_dtype = np.float16  # numpy类型
    out_shape = gen_outshape(inputLayout, q_shape, v_shape, numKeyValueHeads, numHeads)



    pfa_param['q_tensor'] = torch_tensor_list[0]
    if flagList[26]:
        pfa_param = deepseek_preprocessing(params, pfa_param, torch_tensor_list, numHeads, numKeyValueHeads,tnd_flag)
    q_tensor, q_bnsd_shape = np_bsh_to_bnsd(pfa_param['q_tensor'], pfa_param['q_shape'], numHeads, actualSeqLengths, inputLayout)
    if inputLayout=="TND" or inputLayout=="NTD":
        pfa_param['lseshape'] = [q_shape[0], q_shape[1], 1]
    else:
        pfa_param['lseshape'] = [q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], 1]
    if 0 in q_shape or len(q_shape) == 0:
        if params["softmax_lse_flag"]:
            return torch.from_numpy(torch_tensor_list[0]),torch.from_numpy(np.array([]))
        return torch.from_numpy(torch_tensor_list[0])





    # >> kv info
    k_tensor_list = []
    v_tensor_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_raw_list = []
    v_shape_bnsd_raw_list = []
    k_start_index = pfa_param['k_start_index']
    k_end_index = pfa_param['k_end_index']
    v_start_index = pfa_param['v_start_index']
    v_end_index = pfa_param['v_end_index']
    kvs_max = 0
    kvs_list = []
    print(f"k_start_index:{k_start_index}, k_end_index:{k_end_index}")


    for i in range(k_shape_num):
        k_shape = pfa_param['k_shape_list'][i]
        if 0 in k_shape or len(k_shape) == 0:
            if params["softmax_lse_flag"]:
                return torch.zeros(out_shape), torch.zeros(pfa_param['lseshape'])
            return torch.zeros(out_shape)
        k_tensor, k_bnsd_shape = np_bsh_to_bnsd(pfa_param['k_tensor_list'][i], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                kv_inputlayout)
        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_raw_list.append(k_bnsd_shape)
        kvs_list.append(k_bnsd_shape[2])
        kvs_max = max(kvs_max,k_bnsd_shape[2])
        if numKeyValueHeads != numHeads:
            print("broadcast k")
            k_tensor, k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_tensor.dtype )
        k_tensor_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
    for i in range(k_shape_num):
        v_shape = pfa_param['v_shape_list'][i]
        v_tensor, v_bnsd_shape = np_bsh_to_bnsd(pfa_param['v_tensor_list'][i], v_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                kv_inputlayout)
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_raw_list.append(v_bnsd_shape)
        if numKeyValueHeads != numHeads:
            print("broadcast v")
            v_tensor, v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor,v_tensor.dtype)
        v_tensor_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
    kvs = kvs_max
    actualprefixKV = 0
    prefix_kvs = 0
    if flagList[21]:
        prefix_k_shape = params['shape_input'][v_end_index+17]
        prefix_v_shape = params['shape_input'][v_end_index+18]
        if len(params['prefix_act_lens'])!=0:
            actualprefixKV = params['prefix_act_lens'][0]

        prefix_k_tensor,prefix_k_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[v_end_index+17], prefix_k_shape, numKeyValueHeads, actualprefixKV,
                                                inputLayout)

        prefix_v_tensor,prefix_v_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[v_end_index + 18], prefix_v_shape, numKeyValueHeads,
                                         actualprefixKV,
                                         inputLayout)
        if numKeyValueHeads != numHeads:
            prefix_k_tensor, prefix_k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, prefix_k_tensor, prefix_k_tensor.dtype)
            prefix_v_tensor, prefix_v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, prefix_v_tensor, prefix_v_tensor.dtype)
        pfa_param ["actualprefixKV"] = actualprefixKV
        pfa_param ["shared_prefix_k"] = prefix_k_tensor
        pfa_param ["shared_prefix_v"] = prefix_v_tensor
        prefix_kvs = prefix_k_bnsd_shape[2]
        if actualprefixKV==0:
            actualprefixKV = prefix_k_bnsd_shape[2]
        kvs += prefix_kvs

    pse_bnsd_tensor = None
    pse_shift_shape = params['shape_input'][v_end_index+1]  ##1nss or bnss
    qs = q_bnsd_shape[2]

    if flagList[3] == 0 or 0 in pse_shift_shape:
        pse_bnsd_tensor = None
    else:
        pse_dtype_torch = np.float16
        pse_shift = None
        pse_shift_random_flag = True
        if action_type in gold:
            pse_shift = torch_tensor_list[3]
        else:
            if 'prandom' in params:
                if params['prandom'] != 0:
                    pse_shift = torch_tensor_list[v_end_index+1]
                else:
                    pse_shift_random_flag = False
            else:
                pse_shift_random_flag = False
            if not pse_shift_random_flag:
                print("pse")
                pse_shift = get_all_alibi(numHeads, pse_shift_shape)
                pse_shift[:, :, qs:, :] = 1
                pse_shift[:, :, :, kvs:] = 1
                npu_pse_shift = torch.from_numpy(pse_shift).to(pse_dtype_torch)
                pse_shift = npu_pse_shift.to(torch.float32).numpy().astype(np.float32)
                tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor', mode='rewrite',
                                              tensors=npu_pse_shift,
                                              params=params)
        cpu_pse_shift = pse_shift[:, :, :qs, :kvs]

        pse_bnsd_tensor = _np_broadcast_pseShift_n(cpu_pse_shift, pse_shift_shape, q_bnsd_shape[0])  # to bnsd
    m_bnsd_tensor = None
    npu_m_shape = params['shape_input'][v_end_index+2]
    m_dtype = np.float16

    randoms = 0
    mrandom_type = "NORMAL"
    if 'mrandomtype' in params:
        mrandom_type = params['mrandomtype']
        if mrandom_type == 'ones':
            randoms = int(params['mrandom'])

    if flagList[4] == 0 or 0 in npu_m_shape:
        preTokens = 214748647
        nextTokens = 214748647
    else:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = npu_m_shape
        if sp_mode == 0 or sp_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(npu_m_shape,q_bnsd_shape)  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        cpu_m_shape = [qs, kvs]  # cpu

        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = _create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
                                                                                         m_dtype, preTokens, nextTokens,
                                                                                         actualSeqLengths,
                                                                                         actualSeqLengthsKV, actualprefixKV,prefix_kvs,kvs_list, batch,
                                                                                         numheads, sp_mode,
                                                                                         random_ones=randoms)
        if mrandom_type == 'invalid' or mrandom_type == 'invaild':
            randoms = int(params['mrandom'])
            cpu_m_tensor[..., :randoms] = 1
            npu_m_tensor[..., :randoms] = 1

        npu_m_tensor = torch.from_numpy(npu_m_tensor.astype(np.int8))
        print("npu_m_tensor shape",npu_m_tensor.shape)
        case_name = params["case_name"]
        torch.save(npu_m_tensor, '{}_attenmask1.pt'.format(case_name))
        align_row_num = 32 - npu_m_tensor.shape[0] % 32 if npu_m_tensor.shape[0] % 32 != 0 else 0
        align_col_num = 32 - npu_m_tensor.shape[1] % 32 if npu_m_tensor.shape[1] % 32 != 0 else 0
        npu_m_tensor = torch.nn.functional.pad(npu_m_tensor, (0, align_col_num, 0, align_row_num), mode='constant', value=1)
        npu_m_tensor_bool = npu_m_tensor > 0
        
        print("npu_m_tensor_bool",npu_m_tensor_bool)
        torch.save(npu_m_tensor_bool, '{}_attenmask.pt'.format(case_name))
        # torch.save(npu_m_tensor, '{}_attenmask1.pt'.format(case_name))
        # tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite', tensors=npu_m_tensor,
        #                               params=params)



        if sp_mode == 0 or sp_mode == 1:
            m_bnsd_tensor = _np_broadcast_mask_n(cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0])
        else:
            m_bnsd_tensor = cpu_m_tensor

    dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = None, None, None, None, None

    if flagList[7]:
        dequant_scale1 = torch_tensor_list[v_end_index+3]
        dequant_scale1 = np.frombuffer(dequant_scale1, dtype=np.float32)
        dequant_scale1 = dequant_scale1[: 1]

    if flagList[8]:
        quant_scale1 = torch_tensor_list[v_end_index+4]
    if flagList[9]:
        dequant_scale2 = torch_tensor_list[v_end_index+5]
        dequant_scale2 = np.frombuffer(dequant_scale2, dtype=np.float32)
        dequant_scale2 = dequant_scale2[: 1]
    if flagList[10]:
        quant_scale2 = torch_tensor_list[v_end_index+6]
        quant_scale2_shape = params['shape_input'][v_end_index+6]
        per_channel = True
        if len(quant_scale2_shape) == 1:
            if quant_scale2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_scale2 = _trans_1h_to_1n1d(quant_scale2_shape, quant_scale2,
                                             numHeads, inputLayout)
    if flagList[11]:
        quant_offset2 = torch_tensor_list[v_end_index+7]
        quant_offset2_shape = params['shape_input'][v_end_index+7]
        per_channel = True
        if len(quant_offset2_shape) == 1:
            if quant_offset2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_offset2 = _trans_1h_to_1n1d(quant_offset2_shape, quant_offset2,
                                              numHeads, inputLayout)


    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    anti_or_kvanti = 0
    if len(flagList)>20:
        # 获取kv量化参数
        if flagList[12]:
            anti_or_kvanti = 1
            antiquantscale_shape = params['shape_input'][v_end_index+8]
            antiquantscale_tensor = torch_tensor_list[v_end_index+8]
            if len(antiquantscale_shape) == 1:  #pertensor
                print("antiquantscale_pertensor")
                antiquant_scale = antiquantscale_tensor
            else:  #perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                antiquantscale_2n1d_tensor = _trans_2h_to_2n1d(antiquantscale_shape, antiquantscale_tensor, numKeyValueHeads)

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads != numHeads:
                    print("broadcast kvanti")
                    antiquant_scale = broadcast_kv_dequant_tensor(antiquantscale_2n1d_tensor, numKeyValueHeads, numHeads)
                else:
                    antiquant_scale = antiquantscale_2n1d_tensor
        if flagList[13]:
            antiquantoffset_shape = params['shape_input'][v_end_index+9]
            antiquantoffset_tensor = torch_tensor_list[v_end_index+9]
            if len(antiquantoffset_shape) == 1:
                print("antiquantoffset_pertensor")
                antiquant_offset = antiquantoffset_tensor
            else:
                antiquantoffset_2n1d_tensor = _trans_2h_to_2n1d(antiquantoffset_shape, antiquantoffset_tensor,numKeyValueHeads)

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads!=numHeads:
                    print("broadcast kvanti")
                    antiquant_offset = broadcast_kv_dequant_tensor(antiquantoffset_2n1d_tensor, numKeyValueHeads, numHeads)
                else:
                    antiquant_offset = antiquantoffset_2n1d_tensor
        # 判断kv分离的伪量化

        if flagList[17] or flagList[19]:
            anti_or_kvanti = 2
            q_dtype_np = np.float16
            k_antiquantscale_shape = params['shape_input'][v_end_index + 13]
            k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
            v_antiquantscale_shape = params['shape_input'][v_end_index + 15]
            v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]

            if params['k_antiquant_mode'] ==0:
                if len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1:  # pertensor
                    print("antiquantscale_pertensor")
                    k_antiquant_scale = k_antiquantscale_tensor
                    v_antiquant_scale = v_antiquantscale_tensor
                else:  # perchanel
                    # 将kv量化参数转换为n1d(匹配bnsd)
                    k_antiquantscale_n1d_tensor = _trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,numKeyValueHeads)
                    v_antiquantscale_n1d_tensor = _trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,numKeyValueHeads)
                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        print("broadcast kvanti")
                        k_antiquant_scale = broadcast_kv_split_dequant_tensor(k_antiquantscale_n1d_tensor, numKeyValueHeads,numHeads,q_dtype_np)
                        v_antiquant_scale = broadcast_kv_split_dequant_tensor(v_antiquantscale_n1d_tensor,numKeyValueHeads,numHeads,q_dtype_np)
                        print(v_antiquant_scale.shape)
                    else:
                        k_antiquant_scale = k_antiquantscale_n1d_tensor
                        v_antiquant_scale = v_antiquantscale_n1d_tensor
            if params['k_antiquant_mode'] ==1:#pertoken
                # pertoken -> BS
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
        pfa_param['k_antiquant_scale'] = k_antiquant_scale
        pfa_param['v_antiquant_scale'] = v_antiquant_scale

        if flagList[18] or flagList[20]:
            q_dtype_np = np.float16
            k_antiquantoffset_shape = params['shape_input'][v_end_index + 14]
            k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
            v_antiquantoffset_shape = params['shape_input'][v_end_index + 16]
            v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

            if params['k_antiquant_mode'] ==0:
                if len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1:  # pertensor
                    print("antiquantscale_pertensor")
                    k_antiquant_offset = k_antiquantoffset_tensor
                    v_antiquant_offset = v_antiquantoffset_tensor
                else:  # perchanel
                    # 将kv量化参数转换为2n1d(匹配bnsd)
                    k_antiquantoffset_n1d_tensor = _trans_h_to_n1d(k_antiquantoffset_shape, k_antiquantoffset_tensor,numKeyValueHeads)
                    v_antiquantoffset_n1d_tensor = _trans_h_to_n1d(v_antiquantoffset_shape, v_antiquantoffset_tensor,numKeyValueHeads)

                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        print("broadcast kvanti")
                        k_antiquant_offset = broadcast_kv_split_dequant_tensor(k_antiquantoffset_n1d_tensor, numKeyValueHeads,numHeads,q_dtype_np)
                        v_antiquant_offset = broadcast_kv_split_dequant_tensor(v_antiquantoffset_n1d_tensor,numKeyValueHeads,numHeads,q_dtype_np)
                    else:
                        k_antiquant_offset = k_antiquantoffset_n1d_tensor
                        v_antiquant_offset = v_antiquantoffset_n1d_tensor
            if params['k_antiquant_mode'] ==1:#pertoken
                # pertoken -> BS
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
        pfa_param['k_antiquant_offset'] = k_antiquant_offset
        pfa_param['v_antiquant_offset'] = v_antiquant_offset


        pfa_param['anti_or_kvanti'] = anti_or_kvanti
        pfa_param['k_antiquant_mode'] = params['k_antiquant_mode']
        if flagList[14]:
            # shape_input取tensorshape
            blockSize = params['blocksize']
            btShape = params['shape_input'][v_end_index+10]
            kvcacheShape = params['shape_input'][v_end_index+19]
            if 0 in btShape:
                print('[WARNING]:block_table为空场景，输出空tensor！')
                return torch.zeros(out_shape)
            if 0 in kvcacheShape:
                print('[WARNING]:PA场景下kvcache为空tensor，输出空tensor！')
                return torch.zeros(out_shape)
            block_idx_list = np.arange(kvcacheShape[0])
            block_idx_list = np.random.permutation(block_idx_list)
            block_idx = 0
            block_table = np.full(btShape,-1).astype(np.int32)
            blockNumPerBlock = []
            for actual_seq in actualSeqLengthsKV:
                blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            for idx,block_per_b in enumerate(blockNumPerBlock):
                for j in range(block_per_b):
                    block_table[idx][j] = (block_idx_list[block_idx])
                    block_idx += 1
            tools.modify_alcnn_input_file(ids=12, origin_index=[12], type='tensor', mode='rewrite',
                                          tensors=torch.from_numpy(block_table),
                                          params=params)

            k_cache = torch_tensor_list[v_end_index + 19]
            v_cache = torch_tensor_list[v_end_index + 20]
            # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
            if k_cache.ndim==3:
                k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_shape_bnsd_raw_list[0])
                v_tensor_bsh_raw = trans_bnsd_to_bsh(v_tensor_bnsd_list[0], v_shape_bnsd_raw_list[0])
                for b,block_per_b in enumerate(blockNumPerBlock):
                    for blockid_index in range(block_per_b):
                        blockid = block_table[b][blockid_index]
                        block_offset = blockid_index * blockSize
                        if blockid_index == block_per_b - 1:
                            blocksize_left = actualSeqLengthsKV[b]-block_offset
                            k_cache[blockid, 0:blocksize_left, :] = k_tensor_bsh_raw[b,
                                                               block_offset:(block_offset + blocksize_left), :]
                            v_cache[blockid, 0:blocksize_left, :] = v_tensor_bsh_raw[b,
                                                               block_offset:(block_offset + blocksize_left), :]
                        else:
                            k_cache[blockid, 0:blockSize, :] = k_tensor_bsh_raw[b,block_offset:(block_offset + blockSize), :]
                            v_cache[blockid, 0:blockSize, :] = v_tensor_bsh_raw[b,block_offset:(block_offset + blockSize), :]
            if k_cache.ndim==4:
                k_tensor_bsh_raw = k_tensor_bnsd_list[0]
                v_tensor_bsh_raw = v_tensor_bnsd_list[0]

                # gen kv cache
                for b, block_per_b in enumerate(blockNumPerBlock):
                    for blockid_index in range(block_per_b):
                        blockid = block_table[b][blockid_index]
                        block_offset = blockid_index * blockSize
                        if blockid_index == block_per_b-1:
                            blocksize_left = actualSeqLengthsKV[b]-block_offset
                            k_cache[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[b, :, block_offset:(block_offset + blocksize_left), :]
                            v_cache[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[b, :,block_offset:(block_offset + blocksize_left), :]
                        else:
                            k_cache[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[b,:,block_offset:(block_offset + blockSize), :]
                            v_cache[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[b,:,block_offset:(block_offset + blockSize), :]




            kv_dtype = np.float16

            if str(kv_dtype) == "<class 'bfloat16'>":
                k_cache = torch.from_numpy(k_cache).to(torch.bfloat16)
                v_cache = torch.from_numpy(v_cache).to(torch.bfloat16)
            elif str(kv_dtype) != "<class 'float8_e5m2'>" and params['dtype_input'][1] != "hifloat8" and params['dtype_input'][1] != "float8_e4m3fn":
                k_cache = torch.from_numpy(k_cache.astype(kv_dtype))
                v_cache = torch.from_numpy(v_cache.astype(kv_dtype))

            if str(kv_dtype) != "<class 'float8_e5m2'>" and params['dtype_input'][1] != "hifloat8" and params['dtype_input'][1] != "float8_e4m3fn":
                tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list', mode='rewrite', tensors=[k_cache],
                                            params=params)
                tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list', mode='rewrite', tensors=[v_cache],
                                            params=params)
            else:
                tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list',
                                            mode='rewrite',
                                            tensors=[k_cache],
                                            params=params,
                                            data_dtype=params['dtype_input'][1])
                tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list',
                                            mode='rewrite',
                                            tensors=[v_cache],
                                            params=params,
                                            data_dtype=params['dtype_input'][2])



    if flagList[15]:
        pfa_param['queryPaddingSize'] = torch_tensor_list[v_end_index+11]
    if flagList[16]:
        pfa_param['kvPaddingSize'] = torch_tensor_list[v_end_index+12]
    pfa_param['lseflag'] = params["softmax_lse_flag"]
    pfa_param['lseshape'] = [q_bnsd_shape[0],q_bnsd_shape[1],q_bnsd_shape[2],1]


    y_all,lse = _np_promtattention_bnsd(q_tensor, q_bnsd_shape, k_tensor_list, k_shape_bnsd_list, v_tensor_list, v_shape_bnsd_list,
                                    pse_bnsd_tensor,
                                    m_bnsd_tensor, scaleValue, actualSeqLengths, actualSeqLengthsKV, preTokens,
                                    nextTokens, sp_mode, pfa_param, dequant_scale1,
                                    dequant_scale2, quant_scale1, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, out_dtype, params['dtype_input'][0])
    if inputLayout == "BSH":
        y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape)
    elif inputLayout == "NSD":
        y_all = y_all.reshape(out_shape)
    elif inputLayout == "BSND" or inputLayout == "BNSD_BSND":
        y_all = y_all.transpose(0, 2, 1, 3)
    elif inputLayout == "TND":
        T = sum(actualSeqLengths)
        B = len(actualSeqLengths)
        N = out_shape[1]
        D = out_shape[2]
        output = np.zeros((T, N, D), dtype=y_all.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actualSeqLengths[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
            t_start += act_s
        y_all = output
    elif len(out_shape) == 2:
        if q_bnsd_shape[0] == 1:
            y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape).astype(np.float32)
        else:
            sums = 0
            B = q_bnsd_shape[0]
            N = q_bnsd_shape[1]
            S = q_bnsd_shape[2]
            D = q_bnsd_shape[3]
            y_all = y_all.transpose(0, 2, 1, 3)
            yNewAll = np.zeros((S, N * D))
            for i in range(B):
                for j in range(actualSeqLengths[i]):
                    yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                sums += actualSeqLengths[i]
            y_all = yNewAll.astype(np.float32)
    else:
        y_all = y_all
    y_all = torch.from_numpy(y_all)
    if params["softmax_lse_flag"]==True:
        if inputLayout == "TND":
            T = sum(actualSeqLengths)
            B = len(actualSeqLengths)
            N = out_shape[1]
            lse_output = np.zeros((T, N, 1), dtype=lse.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    lse_output[t_start:t_end, n_index, :] = lse[b_index, n_index, :act_s, :]
                t_start += act_s
            lse = lse_output
        lse = torch.from_numpy(lse.astype(np.float32))
        return y_all,lse

    # lse = np.zeros([1],dtype=np.float32)
    # # lse = np.zeros(pfa_param['lseshape'], dtype=np.float32)
    # lse = torch.from_numpy(lse.astype(np.float32))
    # return y_all,lse
    return y_all

def aclnn_op_func(torch_tensor_list, params):
    action_type = params["action_type"]
    cfg_fk = tools.ConfigFmk()

    if action_type in ("bm", "bm_output", "multi_bm", "bm_gold", "bm_output_gold"):
        if cfg_fk.device_type in ("gpu",):
            device_type = "gpu"
        else:
            device_type = "cpu"
    else:
        device_type = "npu"
    print("device_type", device_type)

    q_shape = params['shape_input'][0]
    print(params)
    inputLayout = params['inputlayout']
    if inputLayout == "TND":  # TND格式
        qtensor_length = len(params["actualseqlengths"])
    else:
        qtensor_length = qtensor_seqlength(q_shape, inputLayout)

    if device_type == "cpu":
        if qtensor_length == 1:
            print("ifa","*"*30)
            return ifa.aclnn_op_func_ifa_cpu(torch_tensor_list, params)
        else:
            print("pfa","*"*30)
            pfa_cpu_golden = aclnnPromptFlashAttention(torch_tensor_list, params)
            return pfa_cpu_golden

    elif device_type == "gpu":
        print(f"gpu 执行中...")
        # enable_gpu = 1 #先默认是true,走大算子
        device_id = os.environ['DEVICE_ID']
        print("device_id is ", device_id)
        torch.cuda.set_device(f"cuda:{device_id}")
        if qtensor_length == 1:
            print("ifa", "*" * 30)
            return ifa.aclnn_op_func_ifa_gpu(torch_tensor_list, params)
        else:
            if "fused" in params:
                return ifa.aclnn_op_func_ifa_gpu(torch_tensor_list, params)
            else:
                print("pfa", "*" * 30)
                pfa_gpu_golden = aclnnPromptFlashAttention_GPU(torch_tensor_list, params)
                return pfa_gpu_golden

    else:
        print(f"npu 执行中...")
        if qtensor_length == 1:
            #走ifa逻辑
            print("ifa", "*" * 30)
            return pfa.aclnnPromptFlashAttention_GPU_unification(torch_tensor_list, params)
        else:
            pfa_npu_result = aclnnPromptFlashAttention_NPU(torch_tensor_list, params)
            return pfa_npu_result


def get_operator(input_tensor_list, params, device, device_id):
    if device == "npu":
        torch.npu.set_device(f"npu:{device_id}")
    torch_tensor_list = list()
    for i in range(len(input_tensor_list)):
        input_tensor = input_tensor_list[i]

        if any("Tensor" in _key for _key in params.keys()):
            if device == "cpu" and input_tensor.dtype == torch.float16:
                input_tensor = input_tensor.float()
            torch_tensor_list.append(input_tensor.to(device))
        else:
            if device == "cpu":
                # print(i)
                # print(input_tensor)
                if input_tensor.dtype == np.float16 or str(input_tensor.dtype) == "bfloat16":
                    input_tensor = input_tensor.astype(np.float32)
                torch_tensor_list.append(input_tensor)
            if device in ["npu", "cuda"]:
                if str(input_tensor.dtype) == "uint64":
                    print(type(input_tensor), "torch不支持uint64！")
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.int64)).to(torch.int64)
                    torch_tensor_list.append(input_tensor_i.to(device))

                elif str(input_tensor.dtype) == "bfloat16":
                    input_tensor_i = torch.from_numpy(input_tensor.astype(np.float32)).to(torch.bfloat16)
                    torch_tensor_list.append(input_tensor_i.to(device))
                else:
                    input_tensor_i = torch.from_numpy(input_tensor)
                    torch_tensor_list.append(input_tensor_i.to(device))

    if device == "npu":
        output_data = training.aclnn_execute_npu(torch_tensor_list, params)
    elif device == "npu_off":
        training.aclnn_execute_npu_off(torch_tensor_list, params)
        return
    else:
        output_data = training.aclnn_execute_cpu(torch_tensor_list, params)

    if isinstance(output_data, list) or isinstance(output_data, tuple):
        output_value = []
        get_output_list(True, output_data, output_value)
        output_list = []
        for i in range(len(output_value)):
            tensor_tmp = output_value[i].to("cpu")
            if params["dtype_output"][i] == "bf16":
                tensor_tmp = tensor_tmp.float()
            tensor_tmp = tensor_tmp.detach().numpy()
            if device == "cpu" and np.float16 == np.float16:
                tensor_tmp = tensor_tmp.astype(np.float16)
            output_list.append(tensor_tmp)
        return output_list
    elif isinstance(output_data, bool):
        output_data = output_data
    else:
        output_data = output_data.to("cpu")
        if params["dtype_output"][0] == "bf16":
            output_data = output_data.float()
        output_data = output_data.detach().numpy()
        if device == "cpu" and params["dtype_output"][0] and np.float16 == np.float16:
            output_data = output_data.astype(np.float16)

    return output_data

def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i+1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0")
    print(f"[INFO]before trans: {list}")
    print(f"[INFO]after trans: {list_new}")
    return list_new

# @set_timeout(120)
def aclnnFusedInferAttentionScoreV3(action_type, params, case_path, op_name, soc_version, framework):
    case_name = params["case_name"]
    params["op_name"] = op_name
    params["case_path"] = case_path
    result, error_percent, max_error = "Pass", 100.0, 1.0
    if action_type in ("bm", "one"):
        tools.print("-----------------------Start to generate cpu pytorch atenIR golden--------")
        result, error_percent, max_error = training.run_pytorch_training_cpu(params)
    if action_type in ("bm_input",):
        tools.print("-----------------------Start to run bm_input-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_input(params)
    if action_type in ("bm_output",):
        tools.print("-----------------------Start to run bm_output-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_output(params)
    if action_type in ("bm_gold",):
        tools.print("-----------------------Start to run bm_gold-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_gold(params)
    if action_type in ("bm_output_gold",):
        tools.print("-----------------------Start to run bm_output_gold-------")
        result, error_percent, max_error = training.run_pytorch_training_bm_output_gold(params)
    if action_type in ("npu", "one") and result == "Pass":
        tools.print("-----------------------Start to run aclnn online--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_online(params)
    if action_type in ("npu_off",) and result == "Pass":
        tools.print("-----------------------Start to run aclnn offline--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_offline_two_input(params)
    if action_type in ("npu_off_all",) and result == "Pass":
        if params['index'] == 0:
            tools.print("-----------------------Start to run aclnn offline all--------")
        result, error_percent, max_error = training.run_pytorch_training_npu_offline_all_two_input(params)
    if action_type in ("compare",) and result == "Pass":
        tools.print("-----------------------Start to compare aclnn offline--------")
        result, error_percent, max_error = training.run_pytorch_training_compare_two_input(params)
    return result, error_percent, max_error

def npSoftmax_new1(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_max, x_sum

def test(bsh_tensor_list, scale,case_name):
    import torch.nn.functional as F 
    q_shape = bsh_tensor_list[0].numpy()
    k_shape = bsh_tensor_list[1].numpy()
    v_shape = bsh_tensor_list[2].numpy()
    q_shape = q_shape.reshape(*q_shape.shape[:-1],32,64).transpose(0,2,1,3)
    k_shape = k_shape.reshape(*k_shape.shape[:-1],32,64).transpose(0,2,1,3)
    v_shape = v_shape.reshape(*v_shape.shape[:-1],32,64).transpose(0,2,1,3)
    q_shape = torch.from_numpy(q_shape)
    k_shape = torch.from_numpy(k_shape)
    v_shape = torch.from_numpy(v_shape)

    tp= torch.matmul(q_shape.float(), k_shape.transpose(3,2).float()) * scale
    t_mask = torch.load('{}_attenmask1.pt'.format(case_name))
    # t_mask = t_mask.reshape(1,1,27,27)
    # y = np.zeros(q_shape.shape, dtype=np.float32)
    for i in range(32):
        tp_1 = tp[0][i]
        tp_1=tp_1.masked_fill(t_mask == True, -10000.0)
        tp_1, x_max, x_sum = npSoftmax_new1(tp_1.numpy())
        tp_1 = tp_1/x_sum
        # print("tp1 shape",tp_1.shape)
        tp[0][i] = torch.from_numpy(tp_1)
    # tp=F.softmax(tp, dim=-1)
    #print(t_p)
    t_out =torch.matmul(tp.float(), v_shape.float()).numpy().transpose(0,2,1,3)
    return torch.from_numpy(t_out)

if __name__ == "__main__":
    # 输入参数
    # b_list = [198,100,296,149,236,119,142,71,72,300,150,262,131,190,95,234,117,208,104,206,103,250,126,138,69]
    df = pd.read_excel('FIA_test.xlsx', sheet_name=0)
    case_num = len(df['CaseName'])
    for case_index in range(case_num):
        if case_index != 18:
            continue
        inputlayout = df['inputlayout'][case_index]
        sparsemode = df['SparseMode'][case_index]
        ROPE_D = df['ropeD'][case_index]
        nexttokens = df['nextTokens'][case_index]
        pretokens = df['preTokens'][case_index]
        flaglist = eval(df['flagList'][case_index])
        B = df['B'][case_index]
        Q_N = df['Q_N'][case_index]
        KV_N = df['KV_N'][case_index]
        D = df['D'][case_index]
        S = df['S'][case_index]
        value = 1/math.sqrt(D)
        actseqlen = eval(df['actseqlen'][case_index])
        actseqlenkv = actseqlen
        seqlen_max = max(actseqlenkv)

        q_seqlen_sum = 0
        kv_seqlen_sum = 0
        for k in range(B):
            q_seqlen_sum += actseqlen[k]
            kv_seqlen_sum += actseqlenkv[k]

        shape_input = [[B, S, Q_N, D + ROPE_D], [B, S, KV_N, D + ROPE_D], [B, S, KV_N, D], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
        shape_input_no_rope = [[B, S, Q_N, D], [B, S, KV_N, D], [B, S, KV_N, D], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
        shape_input_rope = [[B, S, Q_N, ROPE_D], [B, S, KV_N, ROPE_D]]
        if inputlayout == 'BSH':
            shape_input = [[B, S, Q_N * (D + ROPE_D)], [B, S, KV_N * (D + ROPE_D)], [B, S, KV_N * D], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
            shape_input_no_rope = [[B, S, Q_N * D], [B, S, KV_N * D], [B, S, KV_N * D], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
            shape_input_rope = [[B, S, Q_N * ROPE_D], [B, S, KV_N * ROPE_D]]
        if flaglist[4] == 1:
            shape_input[4] = [S, S]
        case_name  = df['CaseName'][case_index]
        # params = {'inputlayout': inputlayout, 'actualseqlengths': actseqlen, 'actualseqlengthskv': actseqlen, "shape_input": shape_input, 'numheads': Q_N, "scalevalue": 1/math.sqrt(Q_N), "pretokens": 214748647, "nexttokens": 214748647, "numkeyvalueheads": Q_N, "sparsemode": 0, 'dtype_input': ['float16','float16','float16'], 'flaglist': [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0], 'softmax_lse_flag': False}
        params = {'shape_input': shape_input, 'range_input': [[-1, 1], [-1, 1], [-1, 1], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null'], ['null', 'null']], 'dtype_input': ['fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'uint64', 'fp32', 'uint64', 'fp16', 'fp16', 'fp16', 'fp16', 'int32', 'int64', 'int64', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16', 'fp16'], 'format_input': ['ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND', 'ND'], 'dtype_output': ['fp16'], 'type_input': ['tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor', 'tensor'], 'diff_thd': 0.001, 'pct_thd': 0.001, 'max_diff_thd': 0.1, 'rtol': 0.005, 'atol': 2.5e-05, 'bm_cmp_std': {'fp32': {'max_re_rtol': 10.0, 'avg_re_rtol': 2.0, 'rmse_rtol': 2.0, 'small_value': 1e-06, 'small_value_atol': 0.0}, 'fp16': {'max_re_rtol': 10.0, 'avg_re_rtol': 2.0, 'rmse_rtol': 2.0, 'small_value': 0.001, 'small_value_atol': 0.001}, 'bf16': {'max_re_rtol': 10.0, 'avg_re_rtol': 2.0, 'rmse_rtol': 2.0, 'small_value': 1e-07, 'small_value_atol': 0.004}}, 'red_range': {'fp32': '0.000001/0.00001/0.0001/0.0005', 'fp16': '0.001/0.002/0.005/0.01', 'bf16': '0.001/0.002/0.005/0.01', 'hf32': '0.001/0.002/0.005/0.01'}, 'distribution_input': 'uniform', 'attr_1': 'actualseqlengths', 'actualseqlengths': actseqlen, 'required_actualseqlengths': 1, 'attr_2': 'actualseqlengthskv', 'actualseqlengthskv': actseqlen, 'required_actualseqlengthskv': 1, 'attr_3': 'prefix_act_lens', 'prefix_act_lens': [], 'required_prefix_act_lens': 1, 'attr_4': 'numheads', 'numheads': Q_N, 'required_numheads': 1, 'attr_5': 'scalevalue', 'scalevalue': value, 'required_scalevalue': 1, 'attr_6': 'pretokens', 'pretokens': pretokens, 'required_pretokens': 1, 'attr_7': 'nexttokens', 'nexttokens': nexttokens, 'required_nexttokens': 1, 'attr_8': 'inputlayout', 'inputlayout': inputlayout, 'required_inputlayout': 1, 'attr_9': 'numkeyvalueheads', 'numkeyvalueheads': KV_N, 'required_numkeyvalueheads': 1, 'attr_10': 'sparsemode', 'sparsemode': 0, 'required_sparsemode': 1, 'attr_11': 'innerprecise', 'innerprecise': 1, 'required_innerprecise': 1, 'attr_12': 'blocksize', 'blocksize': 0, 'required_blocksize': 1, 'attr_13': 'antiquant_mode', 'antiquant_mode': 0, 'required_antiquant_mode': 1, 'attr_14': 'softmax_lse_flag', 'softmax_lse_flag': False, 'required_softmax_lse_flag': 1, 'attr_15': 'k_antiquant_mode', 'k_antiquant_mode': 0, 'required_k_antiquant_mode': 1, 'attr_16': 'v_antiquant_mode', 'v_antiquant_mode': 0, 'required_v_antiquant_mode': 1, 'attr_17': 'fused_flag', 'fused_flag': 0, 'required_fused_flag': 1, 'attr_18': 'mrandomtype', 'mrandomtype': 'Normal', 'required_mrandomtype': 1, 'attr_19': 'mrandom', 'mrandom': 0, 'required_mrandom': 1, 'attr_20': 'prandom', 'prandom': 0, 'required_prandom': 1, 'attr_21': 'enablegpu', 'enablegpu': 'True', 'required_enablegpu': 1, 'attr_22': 'flaglist', 'flaglist': flaglist, 'required_flaglist': 1, 'case_name': case_name, 'test_type': 'training', 'action_type': 'bm', 'framework_type': 'aclnn', 'device_type': 'npu', 'bin_dir': None, 'precision_method': 0, 'op_name': 'aclnnFusedInferAttentionScoreV3', 'case_path': '/home/wangsong/aclnn_fuzz_0604/libs/../testcase/aclnn_case/aclnnFusedInferAttentionScoreV3'}
        print("params",params)
        q_bsh_num = np.random.uniform(-1, 1, shape_input_no_rope[0]).astype(np.float16)
        k_bsh_num = np.random.uniform(-1, 1, shape_input_no_rope[1]).astype(np.float16)
        v_bsh_num = np.random.uniform(-1, 1, shape_input_no_rope[2]).astype(np.float16)
        if ROPE_D == 64:
            q_rope_num = np.random.uniform(-1, 1, shape_input_rope[0]).astype(np.float16)
            k_rope_num = np.random.uniform(-1, 1, shape_input_rope[1]).astype(np.float16)
            q_rope = torch.tensor(q_rope_num)
            k_rope = torch.tensor(k_rope_num)
            q_attach_num = np.concatenate((q_bsh_num, q_rope_num),axis=-1)
            k_attach_num = np.concatenate((k_bsh_num, k_rope_num),axis=-1)
        else:
            q_rope = None
            k_rope = None
            q_attach_num = q_bsh_num
            k_attach_num = k_bsh_num
        q_bsh = torch.tensor(q_bsh_num)
        k_bsh = torch.tensor(k_bsh_num)
        v_bsh = torch.tensor(v_bsh_num)
        bsh_tensor_list = [q_bsh, k_bsh, v_bsh, q_rope, k_rope]
        numpy_tensor_list=[q_attach_num.astype(np.float32),k_attach_num.astype(np.float32),v_bsh_num.astype(np.float32)]
        
        cpu_out = aclnnPromptFlashAttention_unification(numpy_tensor_list, params)
        # cpu_out1 = test(bsh_tensor_list, value,case_name)
        out_bsh = aclnnPromptFlashAttention_NPU_unification(bsh_tensor_list, params)
        data_compare_np(params, out_bsh.cpu().numpy(), cpu_out.numpy(), 'fp16', 0.005, 0.005, 100,
                0,
                0.005, 0.000025)
        if flaglist[4] == 1:
            os.remove('{}_attenmask.pt'.format(case_name))
