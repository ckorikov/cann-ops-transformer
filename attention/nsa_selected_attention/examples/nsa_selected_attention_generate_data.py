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

import numpy as np
import sys

case_name = sys.argv[1]
if case_name == 'test_nsa_selected_attention':
    query = np.random.uniform(-0.1, 0.1, (1024, 16, 192)).astype(np.float16)
    key = np.random.uniform(-0.1, 0.1, (4096, 4, 192)).astype(np.float16)
    value = np.random.uniform(-0.1, 0.1, (4096, 4, 128)).astype(np.float16)
    topk_indices = np.random.randint(0, 32, (1024, 4, 16)).astype(np.int32)
    query.tofile('query.bin')
    key.tofile('key.bin')
    value.tofile('value.bin')
    topk_indices.tofile('topk_indices.bin')
else:
    raise RuntimeError(f"Invalid case name:", case_name)
