import numpy as np
import torch, torch_npu
import time


def check_diff(x, y):
    diff = np.abs(x - y)
    print(f'Max diff: {diff.max():.04f}')

def profiling(model, inputs, outputs):
    repeat = 10
    experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLeval.Level2)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        experimental_config = experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'./{type}_profile_results')
    ) as prof:
        
        outputs = model(*inputs)

        torch.npu.synchronize()
        t1 = torch.npu.Event(enable_timing=True)
        t2 = torch.npu.Event(enable_timing=True)
        t1.record()
        for _ in range(repeat):
            outputs = model(*inputs)
        t2.record()
        torch.npu.synchronize()
        elapsed = t1.elapsed_time(2)

        print(f'>>>> {type} IMPL TIME ELAPSED: {(elapsed/repeat*1000):.1f} us')

        prof.step()

    return outputs