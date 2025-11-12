/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class QuantReduceScatterTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantReduceScatterTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantReduceScatterTiling TearDown" << std::endl;
    }
};
// 红线用例
TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_critical_case_1)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    uint64_t expectTilingKey = 0UL;
    std::string expectTilingData = "1024 5120 80 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 43;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_critical_case_2)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{2048, 5120}, {2048, 5120}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            {{{2048, 40}, {2048, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{256, 5120}, {256, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT))},
        }, &compileInfo, "Ascend910_95");
    uint64_t expectTilingKey = 0UL;
    std::string expectTilingData = "2048 5120 40 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 43;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_critical_case_3)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 7168}, {1024, 7168}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1024, 56}, {1024, 56}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    uint64_t expectTilingKey = 0UL;
    std::string expectTilingData = "1024 7168 56 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 43;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

// 异常用例——mx量化场景
TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_1)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{0, 5120}, {0, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{0, 80, 2}, {0, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_2)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 0}, {1024, 0}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 0, 2}, {1024, 0, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_3)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 64, 2}, {1024, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_4)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{2, 1024, 5120}, {2, 1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 1024, 80, 2}, {2, 1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_5)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_6)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_7)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 160}, {1024, 160}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_8)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_9)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1023, 80, 2}, {1023, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_10)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 79, 2}, {1024, 79, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_11)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 3}, {1024, 80, 3}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_12)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_13)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_14)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("add")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_15)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_16)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}
// 异常用例——TG量化
TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_17)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{2, 1024, 5120}, {2, 1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 40}, {1024, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_18)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 40}, {1024, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_19)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{{1024, 40}, {1024, 40}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_20)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 40, 2}, {1024, 40, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_21)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 40}, {1024, 40}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_22)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 40}, {1024, 40}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_23)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1023, 40}, {1023, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_24)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 35}, {1024, 35}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_25)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1023, 5120}, {1023, 5120}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1023, 40}, {1023, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_95");
    ExecuteTestCase(tilingContextPara);
}

TEST_F(QuantReduceScatterTiling, quant_reduce_scatter_tiling_abuse_case_26)
{
    struct QuantReduceScatterCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara("QuantReduceScatter", {
            {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{1024, 80, 2}, {1024, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
        }, {
            {{{128, 5120}, {128, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }, {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
        }, &compileInfo, "Ascend910_93");
    ExecuteTestCase(tilingContextPara);
}