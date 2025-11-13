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
#include "mc2_tiling_case_executor.h"

class MatmulAllReduceTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceTiling TearDown" << std::endl;
    }
};

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_1)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 65536UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_1_cube)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_unaligned)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{1, 65536}, {1, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{65536, 128}, {65536, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 128}, {1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_big_N)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 0xFFFFFFF}, {1536, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 65536UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_big_K)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 0xFFFFFFF}, {8192, 0xFFFFFFF}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0xFFFFFFF, 12288}, {0xFFFFFFF, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 65536UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_910B_1)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{12290, 15360}, {12290, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 12288}, {15360, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{12290, 12288}, {12290, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_910B_win2win)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{12290, 15360}, {12290, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 12288}, {15360, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{12290, 12288}, {12290, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_2)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_3)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{128, 1536}, {128, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_4)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{1024, 1536}, {1024, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_5)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_support_3_dim)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{1, 8192, 1536}, {1, 8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 8192, 12288}, {1, 8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_bfloat16)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000001100UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_int8_1)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_int8_2)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 16UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_int8_bf16)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_a8w8_910b)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_a8w8_scaleDimNum2_910b)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8192}, {1, 8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1, 8192}, {1, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 8192}, {1, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_a8w8_910b_mCut_1)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{4096, 6272}, {4096, 6272}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{6272, 8192}, {6272, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_a8w8_910b_mCut_2)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{4096, 1024}, {4096, 1024}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1024, 8192}, {1024, 8192}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 8192}, {4096, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_empty_k)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0, 8192}, {0, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 10000000000000000009UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

//===============================================Ascend310P====================================================
TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_1_310P)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 12288}, {1536, 12288}}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_1_310P_weightnz_antiquant)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 1536}, {8192, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{48, 768, 16, 32}, {48, 768, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_1)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{5512, 11136}, {5512, 11136}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{11136, 4096}, {11136, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5512, 4096}, {5512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_1_k_zero)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{5512, 0}, {5512, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0, 4096}, {0, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5512, 4096}, {5512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2100000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_2)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8198, 2536}, {8198, 2536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2536, 12288}, {2536, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8198, 12288}, {8198, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000001UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_3)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{15356, 5120}, {15356, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 8192}, {5120, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{15356, 8192}, {15356, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000001UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_4)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 30720}, {8192, 30720}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{30720, 2048}, {30720, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 2048}, {8192, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_5)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{128, 40960}, {128, 40960}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{40960, 128}, {40960, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_6)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{8192, 3072}, {8192, 3072}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3072, 8192}, {3072, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2000001UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_mcut_float16_310P_Weight_NZ)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{5512, 11136}, {5512, 11136}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{11136, 4096}, {11136, 4096}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5512, 4096}, {5512, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 67536UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_a8w8_310p)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 8;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 1536}, {256, 1536}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1536, 8192}, {1536, 8192}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 1UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MatmulAllReduceTiling, matmul_all_reduce_test_tiling_float16_empty_k_310p_nz)
{
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MatmulAllReduce",
        {
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{0, 8192}, {0, 8192}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{256, 8192}, {256, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo, "Ascend310P", coreNum, ubSize);    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 2100000UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
