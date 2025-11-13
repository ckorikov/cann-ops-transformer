import torch
import torch_npu
import npu_ops_transformer_ext


total_case_num = 500
rows_case_num = 100
rows_cases = torch.randint(1, 16385, (rows_case_num, ))
ks = torch.randint(1, 17, (total_case_num // rows_case_num, ))
supported_dtypes = {torch.bfloat16}
for data_type in supported_dtypes:
    for rows in rows_cases:
        for k in ks:
            print(f"DataType = <{data_type}>, rows = {rows}, k = {k}")
            x = torch.randn((rows, k)).to(data_type)
            # print(f"Input x: \n{x}")
            golden = (x / x.sum(-1).unsqueeze(-1))*2.5
            print(f"cpu output: \n{golden}")
            x_npu = x.npu()
            torch.ops.npu_ops_transformer_ext.score_normalize(x=x_npu, rows=rows, k=k)
            npu_result = x_npu.cpu()
            print(f"[OK] torch.ops.npu_ops_transformer_ext.score_normalize<{data_type}> successfully!")
            print(f"npu output: \n{npu_result}")
            rotl, atol = torch.testing._comparison.get_tolerances(data_type, rtol=None, atol=None)
            print(f"compare CPU Result vs NPU Result: {torch.allclose(golden, npu_result, rtol = rotl, atol = atol)}\n\n")
