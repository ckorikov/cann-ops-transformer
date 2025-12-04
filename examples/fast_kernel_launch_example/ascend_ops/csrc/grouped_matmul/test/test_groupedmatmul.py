#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import ascend_ops

supported_dtypes = {torch.float16, torch.bfloat16, torch.float}
for data_type in supported_dtypes:
    print(f"DataType = <{data_type}>")
    x = torch.randn(40, 10000).to(data_type)
    print(f"Tensor x = {x}")
    cpu_result = torch.isfinite(x)
    print(f"cpu: isfinite(x) = {cpu_result}")
    x_npu = x.npu()
    npu_result = torch.ops.ascend_ops.isfinite(x_npu).cpu()
    print(f"[OK] torch.ops.ascend_ops.isfinite<{data_type}> successfully!")
    print(f"npu: isfinite(x) = {npu_result}")
    print(f"compare CPU Result vs NPU Result: {torch.allclose(cpu_result, npu_result)}\n\n")

# test reduce-overhead
import torchair
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, x):
        x = torch.ops.ascend_ops.isfinite(x)
        return x

model = SimpleNet().npu()
compiled_model = torch.compile(model, dynamic=False, fullgraph=False, backend=npu_backend)
tensor_input = torch.randn(80, 10).to(torch.float).npu()
tensor_inputs = [torch.randn_like(tensor_input) for _ in range(10)]

def run():
    import time

    eager_time = []
    for data in tensor_inputs:
        torch_npu.npu.synchronize()
        start_time = time.time()
        npu_result = model(data)
        torch_npu.npu.synchronize()
        end_time = time.time()
        eager_time.append(end_time - start_time)

    graph_time = []
    for data in tensor_inputs:
        torch_npu.npu.synchronize()
        start_time = time.time()
        npu_result = compiled_model(data)
        torch_npu.npu.synchronize()
        end_time = time.time()
        graph_time.append(end_time - start_time)

    # summary
    avg_eager_time = sum(eager_time) / len(eager_time)
    avg_graph_time = sum(graph_time) / len(graph_time)
    speedup = avg_eager_time / avg_graph_time
    print(f"Average Eager Mode Duration: {avg_eager_time:.6f} sec.")
    print(f"Average ReduceOverhead Mode Duration: {avg_eager_time:.6f} sec.")
    print(f"Speedup: {speedup:.2f}x")

print("warm up ...")
run()
run()
run()

print("start profiling ...")
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
) as prof:
    for step in range(3):
        run()
        prof.step()
print("done.")
