# GroupedMatMulAlltoAllv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。
- <term>昇腾910_95 AI处理器</term>。

## 功能说明

-   **算子功能**：完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合，**先计算后通信**。
-   **计算公式**：
    -   路由专家：
        $$
            gmmY = gmmX \times gmmWeight \\
            unpermuteOut = Unpermute(gmmY) \\
            y = AlltoAllv(unpermuteOut)
        $$
    
    -   共享专家：
        $$
            mmY = mmX \times mmWeight
        $$



## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 800px">
<col style="width: 800px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>gmmX</td>
<td>输入</td>
<td>该输入进行AlltoAllv通信，通信后结果作为GroupedMatMul计算的左矩阵。支持2维，shape为(A, H1)。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>gmmWeight</td>
<td>输入</td>
<td>GroupedMatMul计算的右矩阵。类型与gmmX保持一致，支持3维，shape为(e, H1, N1)。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>sendCountsTensorOptional</td>
<td>输入</td>
<td>可选输入，当前版本暂不支持，传nullptr。支持shape为(e * epWorldSize,)。</td>
<td>INT32、INT64</td>
<td>ND</td>
</tr>
<tr>
<td>recvCountsTensorOptional</td>
<td>输入</td>
<td>可选输入，当前版本暂不支持，传nullptr。支持shape为(e * epWorldSize,)。</td>
<td>INT32、INT64</td>
<td>ND</td>
</tr>
<tr>
<td>mmXOptional</td>
<td>输入</td>
<td>可选输入，共享专家MatMul计算中的左矩阵。需融合共享专家矩阵计算时该参数必选，支持2维，shape为(BS, H2)。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mmWeightOptional</td>
<td>输入</td>
<td>可选输入，共享专家MatMul计算中的右矩阵。需融合共享专家矩阵计算时该参数必选，支持2维，shape为(H2, N2)。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>group</td>
<td>输入</td>
<td>专家并行的通信域名，字符串长度要求(0, 128)。</td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>epWorldSize</td>
<td>输入</td>
<td>ep通信域size。Atlas A3 训练/推理系列产品支持8、16、32、64；昇腾910_95 AI处理器支持4、8、16、32、64。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sendCounts</td>
<td>输入</td>
<td>表示发送给其他卡的token数，数据类型支持INT64，取值大小为e * epWorldSize，最大为256。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>recvCounts</td>
<td>输入</td>
<td>表示接收其他卡的token数，数据类型支持INT64，取值大小为e * epWorldSize，最大为256。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>transGmmWeight</td>
<td>输入</td>
<td>gmmWeight是否需要转置，true表示需要转置，false表示不转置。</td>
<td>BOOL</td>
<td>ND</td>
</tr>
<tr>
<td>transMmWeight</td>
<td>输入</td>
<td>共享专家mmWeightOptional是否需要转置，true表示需要转置，false表示不转置。</td>
<td>BOOL</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>表示最终计算结果，数据类型与输入gmmX保持一致，支持2维，shape为(BSK, N1)。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mmYOptional</td>
<td>输出</td>
<td>共享专家MatMul的输出，数据类型与mmXOptional保持一致，支持2维，shape为(BS, N2)。仅当传入mmXOptional与mmWeightOptional才输出。</td>
<td>FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
</tbody>
</table>


## 约束说明
- 参数说明里shape使用的变量：
    - BSK：本卡接收的token数，是recvCounts参数累加之和，取值范围(0, 52428800)。
    - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
    - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
    - e：表示单卡上专家个数，e<=32，e * epWorldSize最大支持256。
    - N1：表示路由专家的head_num，取值范围(0, 65536)。
    - N2：表示共享专家的head_num，取值范围(0, 65536)。
    - BS：batch sequence size。
    - K：表示选取TopK个专家，K的范围[2, 8]。
    - A：本卡发送的token数，是sendCounts参数累加之和。
    - ep通信域内所有卡的 A 参数的累加和等于所有卡上的 BSK 参数的累加和。
- 单卡通信量要求在2MB到100MB范围内。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_grouped_mat_mul_allto_allv.cpp](test_grouped_mat_mul_allto_allv.cpp) | 通过[aclnnGroupedMatMulAlltoAllv](aclnnGroupedMatMulAlltoAllv.md)接口方式调用算子。 |