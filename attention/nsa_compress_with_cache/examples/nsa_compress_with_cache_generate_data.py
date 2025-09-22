#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


import numpy as np
import math
import os

np.random.seed(2025)


class Object():
    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.compress_block_size = config["compress_block_size"]
        self.compress_stride = config["compress_stride"]
        self.heads_num = config["heads_num"]
        self.heads_dim = config["heads_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.page_block_size = config["page_block_size"]
        self.result_len = config["result_len"]
        self.block_num_per_batch = self.max_seq_len // self.page_block_size
        self.blocks_num = self.block_num_per_batch * self.batch_size

        self.input_kv = np.random.uniform(0.0, 1.0, [
                                          self.blocks_num, self.page_block_size, self.heads_num, self.heads_dim]).astype(np.float16)
        self.input_weight = np.random.uniform(
            0.0, 1.0, [self.compress_block_size, self.heads_num, 1]).astype(np.float16)
        self.act_seq_lens = np.random.randint(
            0, self.max_seq_len, [self.batch_size]).astype(np.int64)
        self.slot_mapping = np.random.choice(np.arange(
            0, self.result_len), size=self.batch_size, replace=False).astype(np.int32)
        self.input_block_table = np.random.randint(0, self.batch_size * self.block_num_per_batch, [
                                                   self.batch_size, self.block_num_per_batch]).astype(np.int32)
        self.compress_kv_cache = np.random.uniform(
            0.0, 1.0, [self.result_len, self.heads_num, self.heads_dim]).astype(np.float16)
        for i in range(self.batch_size):
            self.act_seq_lens[i] = self.compress_stride * i

    def save_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + "config.txt", "w") as f:
            for key, value in self.config.items():
                f.write(f"{key}:{value}\n")

        self.input_kv = self.input_kv.astype(np.float16)
        self.input_weight = self.input_weight.astype(np.float16)
        self.input_kv.tofile(path + "input_kv.bin")
        self.input_weight.tofile(path + "input_weight.bin")
        self.slot_mapping.tofile(path + "slot_mapping.bin")
        self.act_seq_lens.tofile(path + "act_seq_lens.bin")
        self.input_block_table.tofile(path + "input_block_table.bin")
        self.compress_kv_cache.tofile(path + "compress_kv_cache.bin")

    def print_info(self):
        for key, value in self.config.items():
            print(f"{key}:{value}")

        print(f"kv.shape={self.input_kv.shape}")
        print(f"weight.shape={self.input_weight.shape}")
        print(f"result.shape={self.compress_kv_cache.shape}")

    def get_last_kv(self, batch_idx):
        act_seq_len = self.act_seq_lens[batch_idx]
        block_num = (act_seq_len - 1) // self.page_block_size
        tile_act_seq_len = act_seq_len - block_num * self.page_block_size
        if tile_act_seq_len < self.compress_block_size:
            cur_block_idx = self.input_block_table[batch_idx, block_num]
            pre_block_idx = self.input_block_table[batch_idx, block_num - 1]
            input_kv_cur = self.input_kv[cur_block_idx, 0:tile_act_seq_len]
            input_kv_pre = self.input_kv[pre_block_idx, self.page_block_size -
                                         self.compress_block_size + tile_act_seq_len:self.page_block_size]
            return np.concatenate((input_kv_pre, input_kv_cur), axis=0)

        block_idx = self.input_block_table[batch_idx, block_num]
        return self.input_kv[block_idx, tile_act_seq_len - self.compress_block_size:tile_act_seq_len]

    def calculate_result(self):
        self.input_kv = self.input_kv.astype(np.float32)
        self.input_weight = self.input_weight.astype(np.float32)
        self.compress_kv_cache = self.compress_kv_cache.astype(np.float32)

        compress_last_kv_list = []
        for batch_idx in range(self.batch_size):
            if (self.act_seq_lens[batch_idx] >= self.compress_block_size and (self.act_seq_lens[batch_idx] - self.compress_block_size) % self.compress_stride == 0):
                last_kv = self.get_last_kv(batch_idx)
                weight_last_kv = last_kv * self.input_weight
                compress_last_kv = weight_last_kv.sum(axis=0)
                self.compress_kv_cache[self.slot_mapping[batch_idx]
                                       ] = compress_last_kv

        self.compress_kv_cache = self.compress_kv_cache.astype(np.float16)


def gen_data():
    config = {
        "batch_size": 4,
        "compress_block_size": 32,
        "compress_stride": 16,
        "heads_num": 24,
        "heads_dim": 192,
        "page_block_size": 128,
        "max_seq_len": 512,
        "result_len": 512
    }
    obj = Object(config)
    obj.print_info()
    obj.calculate_result()
    obj.save_data("./data/sample_random/")


if __name__ == "__main__":
    gen_data()
