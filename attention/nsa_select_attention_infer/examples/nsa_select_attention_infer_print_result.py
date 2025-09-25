#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import sys
import multiprocessing


def show():
    case_name = sys.argv[1]
    print_head = "=========================================== Show sample " + case_name + \
                 " result start ================="
    print(print_head)
    if case_name == 'test_nsa_select_attention_infer':
        attention_out = np.fromfile('attention_out.bin', dtype=np.float16)
        print('attentionOut: ', attention_out)

    else:
        raise RuntimeError(f"Invalid case name:", case_name)
    print_end = "=========================================== Show sample " + case_name + \
                " result end ==================="
    print(print_end)


if __name__ == '__main__':
    # 创建进程对象
    p = multiprocessing.Process(target=show)
    # 启动进程
    p.start()
    # 等待进程结束
    p.join()
