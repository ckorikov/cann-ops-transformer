#  FusedInferAttentionScore

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 Training Series Products / Atlas A3 Inference Series Products</term>|      √     |
|<term>Atlas A2 Training Series Products / Atlas 800I A2 Inference Products / A200I A2 Box Heterogeneous Components</term>|      √     |

## 功能说明

- Operator Function: The FlashAttention operator is designed for incremental and full inference scenarios, supporting both full computation scenarios.（[PromptFlashAttention](../prompt_flash_attention/README.md)），It also supports incremental computing scenarios.（[IncreFlashAttention](../incre_flash_attention/README.md)）。

- Calculation formula:

    self-attention（Self-attention constructs an attention model by leveraging the relationships within the input samples themselves. The principle is based on assuming that there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be considered as a token embedding. Applying three weight matrices to such a sequence results in three matrices with dimensions $n \times d$.

    self-attentionThe calculation formula is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input sample, obtained through spatial transformation of the input sample, and can be unified into a single feature space. The term "Attention" in the formula and operator name is an abbreviation for "self-attention."

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the Score function uses the Softmax function, and the self-attention calculation formula is:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    Among these, the product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is typically scaled by dividing by the square root of $d$, and then normalized using softmax on each row. After multiplying with $V$, an $n \times d$ matrix is obtained.

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>parameter name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>input</td>
    <td>Input Q in the formula</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>input</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>input</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>output</td>
    <td>Output in the formula.。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
</tbody>
</table>


## Constraints
- When using this API with PyTorch, ensure that the versions of the CANN-related packages match those of the PyTorch-related packages.

- Handling of empty input parameters: The operator needs to check whether the parameter `query` is empty. If it is empty, the operator returns immediately. If `query` is not an empty tensor, but `key` and `value` are empty tensors (i.e., S2 is 0), then `attentionOut` is filled with zeros. When `attentionOut` is an empty tensor, the AscendCLNN framework will handle it. For other input parameters marked as "can be passed as nullptr" in the parameter description, no processing is performed when these parameters are null pointers.

- The shapes of the corresponding tensors in the `key` and `value` parameters must be exactly the same. In non-contiguous scenarios, the batch size in the tensor list of `key` and `value` can only be 1, and the number of elements in the tensor list must be equal to the B value of `query`. The N and D values of `query` must also be equal. Due to the limitations of the tensor list, the batch size B cannot exceed 256 in non-contiguous scenarios.

- When Q_S is greater than 1, the input of query, key, and value, and the functional usage restrictions are as follows:：
    - Supports B-axis less than or equal to 65536.

  - If the input type is INT8 and the D axis is not 32-byte aligned, the maximum supported value for the B axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the B axis also supports up to 128.
  - The N axis is supported to be less than or equal to 256, and the D axis is supported to be less than or equal to 512. When inputLayout is BSH or BSND, it is recommended that N * D be less than 65535.
  - S is supported to be less than or equal to 20971520 (20M). In some long-sequence scenarios, if the computation workload is too large, it may cause the PFA operator to time out (an AICore error with errorStr: "timeout or trap error"). In such cases, it is recommended to perform S splitting. Note that the computation workload here is affected by B, S, N, and D; the larger these values are, the greater the computation workload. Typical long-sequence scenarios where timeouts may occur (i.e., scenarios where the product of B, S, N, and D is large) include but are not limited to:

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 550px"><colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 150px">
    </colgroup><thead>
    <tr>
    <th>B</th>
    <th>Q_N</th>
    <th>Q_S</th>
    <th>D</th>
    <th>KV_N</th>
    <th>KV_S</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>1</td>
    <td>20</td>
    <td>2097152</td>
    <td>256</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    <tr>
    <td>1</td>
    <td>2</td>
    <td>20971520</td>
    <td>256</td>
    <td>2</td>
    <td>20971520</td>
    </tr>
    <tr>
    <td>20</td>
    <td>1</td>
    <td>2097152</td>
    <td>256</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    <tr>
    <td>1</td>
    <td>10</td>
    <td>2097152</td>
    <td>512</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    </tbody>
    </table>
    </div>

  -  D-axis alignment: When the query, key, value, or attentionOut type includes INT8, the D-axis must be 32-byte aligned; when it includes INT4, the D-axis must be 64-byte aligned; and when all types are FLOAT16 or BFLOAT16, the D-axis must be 16-byte aligned.

- When Q_S equals 1: query, key, and value inputs, the usage restrictions are as follows:

  - Supports B axis ≤ 65536, N axis ≤ 256, and D axis ≤ 512.
  - Currently, this feature is not supported for scenarios where the input types of query, key, and value are all INT8.
  - In the INT4 fake quantization scenario, the aclnn single-operator API supports KV INT4 input or INT4 concatenated into INT32 input (it is recommended to use dynamicQuant to generate INT4 data because dynamicQuant generates an INT32 data type that includes eight INT4 values).
  - In the INT4 fake quantization scenario, if KV INT4 is concatenated into INT32 input, then the N, D, or H of KV is one-eighth of the actual value (the same applies to prefix).
  - There are restrictions on the D axis for key and value under specific data types.
    - When the input types for key and value are INT4 (INT32), the D-axis needs to be 64-byte aligned (INT32 only supports D 8-byte alignment).
  
- The query, key, and value data layout formats can be interpreted from multiple dimensions. Here, B (Batch) represents the batch size of input samples; S (Seq-Length) represents the sequence length of input samples; H (Head-Size) represents the size of the hidden layer; N (Head-Num) represents the number of heads; D (Head-Dim) represents the minimum unit size of the hidden layer, and it satisfies the condition D = H / N; T represents the cumulative sum of the sequence lengths of all batch input samples.

## Usage Instructions


| Invocation Method | Sample Code | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn interface | [test_aclnn_FusedInferAttentionScoreV4](./examples/test_aclnn_fused_infer_attention_score.cpp) | passing through [aclnnFusedInferAttentionScoreV4](./docs/aclnnFusedInferAttentionScoreV4.md) invocation PromptFlashAttentionV3 operator |
