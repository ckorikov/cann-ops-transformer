# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch, torch_npu
import time


def check_diff(x, y):
    diff = torch.abs(x - y)
    rel_diff = diff.max() / torch.abs(x.max())
    print(f'Max diff: {diff.max():.04f} | Relative max diff: {rel_diff:.05f}')

def profiling(model, inputs, type):
    repeat = 10
    experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=1, repeat=1, skip_first=0),
        experimental_config = experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'./{type}_profile_results')
    ) as prof:
        
        outputs = model(*inputs)

        t1 = torch.npu.Event(enable_timing=True)
        t2 = torch.npu.Event(enable_timing=True)
        t1.record()
        for _ in range(repeat):
            outputs = model(*inputs)
        t2.record()
        elapsed = t1.elapsed_time(t2)

        print(f'>>>> {type} IMPL TIME ELAPSED: {(elapsed/repeat*1000):.1f} us')

        prof.step()
