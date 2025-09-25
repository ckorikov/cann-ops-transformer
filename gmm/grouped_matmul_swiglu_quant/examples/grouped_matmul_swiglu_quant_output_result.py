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
import os

if __name__ == '__main__':
    test_case = sys.argv[1]
    print_head = "====================== Show sample " + test_case + " result start ================="
    print(print_head)
    file1 = 'output.bin'
    file2 = 'outputScale.bin'
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        raise RuntimeError(f"Invalid case name:", test_case)
    output = np.fromfile(file1, dtype=np.int8)
    print(f"{test_case} output: ", output)
    outputScale = np.fromfile(file2, dtype=np.float32)
    print(f"{test_case} outputScale: ", outputScale)