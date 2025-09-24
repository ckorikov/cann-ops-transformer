# SwinTransformerLnQKV

## 产品支持情况

<table style="undefined;table-layout: fixed; width: 700px"><colgroup>
<col style="width: 600px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th style="text-align: center;">产品</th>
    <th style="text-align: center;">是否支持</th>
  </tr></thead>
<tbody>
  <tr>
    <td>昇腾910_95 AI处理器</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A3 训练系列产品/Atlas A3 推理系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Atlas 200I/500 A2 推理产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas 推理系列加速卡产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas 训练系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas 200I/300/500 推理产品</td>
    <td style="text-align: center;">×</td>
  </tr>
</tbody>
</table>

## 功能说明

- 算子功能：完成fp16权重场景下的Swin Transformer 网络模型的Q、K、V 的计算。

- 计算公式：

    $$
    (Q,K,V)=((Layernorm(inputX)).transpose() * weight).transpose().split()
    $$  

  其中，weight 是 Q、K、V 三个矩阵权重的拼接。
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
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>inputX</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr> 
  <tr>
    <td>query_output</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key_output</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value_output</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr> 
</tbody>
</table>

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、昇腾910_95 AI处理器：数据类型支持FLOAT16。

## 约束说明
- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor（即S2为0），则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了“可传入nullptr”的入参为空指针时，不进行处理。

- query，key，value输入，功能使用限制如下：

  - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、昇腾910_95 AI处理器：

    - 支持B轴小于等于65536（64k），输入类型包含INT8时D轴非32对齐或输入类型为FLOAT16或BFLOAT16时D轴非16对齐时，B轴仅支持到128。

    - 支持N轴小于等于256。

    - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：
      <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 200px">
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 200px">
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
      
    - 支持D轴小于等于512。inputLayout为BSH或者BSND时，要求N*D小于65535。
    
  - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：在TND场景下query，key，value输入的综合限制：
    - T小于等于65536。
    - N等于8/16/32/64/128，且Q_N、K_N、V_N相等。
    - Q_D、K_D等于192，V_D等于128/192。
    - 数据类型仅支持BFLOAT16。
    - sparse模式仅支持sparse=0且不传mask，或sparse=3且传入mask。
    - 当sparse=3时，要求每个batch单独的actualSeqLengths < actualSeqLengthsKv。
    
  - Atlas 推理系列加速卡产品：
    - 支持B轴小于等于128；支持N轴小于等于256；支持S轴小于等于65535（64k）, Q_S或KV_S非128对齐，Q_S和KV_S不等长的场景不支持配置atten_mask；支持D轴小于等于512。
  
- 当inputLayout为BNSD_BSND时，输入query的shape是BNSD，输出attentionOut的shape为BSND；其余情况attentionOut的shape需要与入参query的shape保持一致。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_PromptFlashAttentionV3](context/PromptFlashAttentionV3.md#调用示例) | 通过[aclnnPromptFlashAttentionV3](../op_host/aclnn_prompt_flash_attention_v3.cpp)调用PromptFlashAttentionV3算子 |
