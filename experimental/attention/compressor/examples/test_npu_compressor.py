# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math, copy
import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
torch.npu.config.allow_internal_format = True

class TestCustomMlaPrologV3(TestCase):
    def test_mla_prolog_v3_pertile_eager(self):
        B = 8
        He = 7168
        S = 1
        D = 128
        block_size = 128
        block_num = 1024
        rope_head_dim = 64
        cmp_ratio = 4
        coff = 2
        norm_eps = 1e-6
        rotary_mode = 1
        T = B * S

        x = torch.rand(T, He, dtype=torch.bfloat16).npu()
        wkv = torch.rand(coff*D, He, dtype=torch.bfloat16).npu()
        wagte = torch.rand(coff*D, He, dtype=torch.bfloat16).npu()
        kv_state = torch.rand(T, coff*cmp_ratio, coff*D, dtype=torch.float32).npu()
        score_state = torch.rand(T, coff*cmp_ratio, coff*D, dtype=torch.float32).npu()
        ape = torch.rand(cmp_ratio, coff*D, dtype=torch.float32).npu()
        norm_weight = torch.rand(D, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(B, S // cmp_ratio, rope_head_dim, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(B, S // cmp_ratio, rope_head_dim, dtype=torch.bfloat16).npu()
        block_table = torch.randint(1, 2, (B, S//block_size), dtype=torch.int32).npu()
        cu_seqlens = torch.randint(1, 2, (B+1,), dtype=torch.int32).npu()
        seqused = torch.randint(1, 2, (B,), dtype=torch.int32).npu()
        start_pos = torch.randint(1, 2, (B,), dtype=torch.int32).npu()

        # start run custom ops
        cmp_kv = (
            torch.ops.custom.npu_compressor(
                x,
                wkv,
                wagte,
                kv_state,
                score_state,
                ape,
                norm_weight, 
                rope_sin,
                rope_cos,
                block_table = block_table,
                cu_seqlens = cu_seqlens,
                seqused = seqused,
                start_pos = start_pos,
                rope_head_dim = rope_head_dim,
                cmp_ratio = cmp_ratio,
                coff = coff,
                norm_eps = norm_eps,
                rotary_mode = rotary_mode
            )
        )


if __name__ == "__main__":
    run_tests()
