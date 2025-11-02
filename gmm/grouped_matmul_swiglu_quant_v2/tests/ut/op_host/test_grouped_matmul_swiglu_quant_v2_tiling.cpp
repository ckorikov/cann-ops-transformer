/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_finalize_routing.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_base_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_fusion_tiling.h"
#include "../../../op_host/op_tiling/grouped_matmul_swiglu_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

class GroupedMatmulSwigluQuantV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRoutingTiling TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulSwigluQuantV2, test_w8a8_normal_case_1)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {192 * 1024 * 1024,
                        24, 48, 128, 256};
    int m = 1024;
    int k = 2048;
    int n = 7168;
    int e = 16;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n / 32, k / 16, 16, 32}, {e, n / 32, k / 16, 16, 32}};
    gert::StorageShape wScaleShape = {{e, n}, {e, n}};
    gert::StorageShape xScaleShape = {{m}, {m}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_INT8, ge::FORMAT_ND},
            {wShape, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
            {wScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 3;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxfp8_normal_case_1)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 2048;
    int k = 7168;
    int n = 4096;
    int e = 8;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {wScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 20000000000;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxfp8_normal_case_2)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 2048;
    int k = 2048;
    int n = 7168;
    int e = 4;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, n, k}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {wShape, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {wScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 20000000001;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxpf8_normal_case_3)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 144;
    int k = 2048;
    int n = 7168;
    int e = 1;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {wShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {wScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 20000000000;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(GroupedMatmulSwigluQuantV2, test_mxpf8_normal_case_4)
{
    optiling::GMMSwigluV2CompileInfo compileinfo = {248 * 1024 * 1024,
                        32, 48, 128, 256, true};
    int m = 288;
    int k = 7168;
    int n = 4096;
    int e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, k, n}, {e, k, n}};
    gert::StorageShape wScaleShape = {{e, k / 64, n, 2}, {e, k / 64, n, 2}};
    gert::StorageShape xScaleShape = {{m, k / 64, 2}, {m, k / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};

    gert::TilingContextPara tilingContextPara("GroupedMatmulSwigluQuantV2", 
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {wShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {wScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {xScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileinfo
    );

    int64_t expectTilingKey = 20000000001;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}