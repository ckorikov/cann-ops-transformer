/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "op_log.h"
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "pad_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/op_tiling_util.h"
#include "common_unittest.h"
#include "transformer/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "fusion_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t ROW_IDX_TYPE_DROPPAD = 0;
constexpr int64_t ROW_IDX_TYPE_DROPLESS = 1;
constexpr int64_t EXPERT_NUM = 256;

class MoeInitRoutingV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingV3Tiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

void RunTestCase(gert::StorageShape x_shape, gert::StorageShape expert_idx_shape, gert::StorageShape *scale_shape,
                 gert::StorageShape *offset_shape, gert::StorageShape expanded_x_shape,
                 gert::StorageShape expanded_row_idx_shape, gert::StorageShape expert_tokens_count_or_cumsum_shape,
                 gert::StorageShape expert_tokens_before_capacity_shape, gert::StorageShape expanded_scale_shape,
                 int64_t activeNum, int64_t C, int64_t expert_num, int64_t dropPadMode, int64_t countFlag,
                 bool tokenFlag, int64_t quantMode, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                 int64_t rowIdxType, ge::graphStatus result, int64_t tilingKey)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 40}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910B"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MoeInitRoutingV3CompileInfo compile_info;

    std::string op_type("MoeInitRoutingV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    ge::DataType expandedXDataType = xDataType;
    if (quantMode != QUANT_MODE_NONE) {
        expandedXDataType = ge::DT_INT8;
    }

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(4, 5)
            .SetOpType(op_type)
            .IrInstanceNum({1, 1, 1, 1})
            .InputShapes({&x_shape, &expert_idx_shape, scale_shape, offset_shape})
            .OutputShapes({&expanded_x_shape, &expanded_row_idx_shape, &expert_tokens_count_or_cumsum_shape,
                           &expert_tokens_before_capacity_shape, &expanded_scale_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, expandedXDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(activeNum)},
                        {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(C)},
                        {"expert_num", ge::AnyValue::CreateFrom<int64_t>(expert_num)},
                        {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                        {"expert_tokens_count_or_cumsum_flag", ge::AnyValue::CreateFrom<int64_t>(countFlag)},
                        {"expert_tokens_before_capacity_flag", ge::AnyValue::CreateFrom<bool>(tokenFlag)},
                        {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(quantMode)},
                        {"acive_expert_range", ge::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                        {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(rowIdxType)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), result);
    if (result == ge::GRAPH_SUCCESS) {
        // todo check tiling result
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKey);
        auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
        std::cout << tiling_data_result << std::endl;
    }
}

void RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag, bool tokenFlag,
                   int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType, std::vector<int64_t> aciveExpertRange,
                   int64_t rowIdxType, ge::graphStatus result, int64_t tilingKey)
{
    gert::StorageShape x_shape = {{N, H}, {N, H}};
    gert::StorageShape expert_idx_shape = {{N, K}, {N, K}};
    gert::StorageShape scale_shape_base = {{N}, {N}};
    gert::StorageShape *scale_shape = nullptr;
    gert::StorageShape *offset_shape = nullptr;
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM;

    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    if (quantMode == QUANT_MODE_DYNAMIC) {
        scale_shape_base = {{E, H}, {E, H}};
        scale_shape = (scaleFlag == 0) ? nullptr : &scale_shape_base;
    } else if (quantMode == QUANT_MODE_STATIC) {
        scale_shape_base = {{E, 1}, {E, 1}};
        if (scaleFlag) {
            scale_shape_base = {{E, H}, {E, H}};
        }
        scale_shape = &scale_shape_base;
    } else {
        scale_shape_base = {{N}, {N}};
        scale_shape = (scaleFlag == 0) ? nullptr : &scale_shape_base;
    }

    gert::StorageShape expert_tokens_count_or_cumsum_shape;
    if (countFlag == 2) { // key-value模式
        expert_tokens_count_or_cumsum_shape = {{E, 2}, {E, 2}};
    } else {
        expert_tokens_count_or_cumsum_shape = {{E}, {E}};
    }

    gert::StorageShape expanded_x_shape = {{N * K, H}, {N * K, H}};
    gert::StorageShape expanded_row_idx_shape = {{N * K}, {N * K}};
    gert::StorageShape expert_tokens_before_capacity_shape = {{E}, {E}};
    gert::StorageShape expanded_scale_shape = {{N * K}, {N * K}};
    RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
                expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, expanded_scale_shape,
                activeNum, C, expert_num, dropPadMode, countFlag, tokenFlag, quantMode, xDataType, aciveExpertRange,
                rowIdxType, result, tilingKey);
}

// 单核 + not quant + scale not None   1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_01)
{
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1000000);
}

// 单核 + not quant + drop pad + scale None 1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_02)
{
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1001000);
}

// 多核 + not quant + scale None 1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_03)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_INT8, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1100000);
}

// 多核 + not quant + drop pad + scale None 1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_04)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000);
}


// 单核 + dynamci quant + scale not None   1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_05)
{
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000);
}

// 单核 + dynamci quant + scale None 1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_06)
{
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1020000);
}

// 单核 + dynamci quant + drop mode + scale not None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_07)
{
    RunNormalCase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000);
}

// 单核 + dynamci quant + drop mode + scale None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_08)
{
    RunNormalCase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1021000);
}

// 多核 + dynamci quant + scale not None  11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_09)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000);
}

// 多核 + dynamci quant + scale None 11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_10)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                  ge::GRAPH_SUCCESS, 1120000);
}

// 多核 + dynamci quant + drop mode + scale not None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_11)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000);
}

// 多核 + dynamci quant + drop mode + scale None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_12)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000);
}

// 多核 + dynamci quant + drop mode + scale None + bfloat16  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_13)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000);
}

// 多核 + dynamci quant + drop mode + scale None + float16 11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_14)
{
    RunNormalCase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1121000);
}

// 单核 + static quant + drop mode + scale not None   1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_15)
{
    RunNormalCase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_FAILED, 1020000);
}

// full load
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_16)
{
    RunNormalCase(1, 7168, 8, 0, 0, 2, true, QUANT_MODE_DYNAMIC, 1, ge::DT_BF16, {0, 256}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 2000000);
}

// performance 单核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_17)
{
    RunNormalCase(1920, 7168, 8, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1201000);
}

// performance 多核gather
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_18)
{
    RunNormalCase(4608, 7168, 8, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1301000);
}

// 多核排序
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_19)
{
    RunNormalCase(4608, 7168, 10, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {0, 8}, ROW_IDX_TYPE_DROPLESS,
                  ge::GRAPH_SUCCESS, 1101000);
}

constexpr int64_t EXPERT_NUM_REGBASE = 200;

void RunTestCaseRegbase(gert::StorageShape x_shape, gert::StorageShape expert_idx_shape,
                        gert::StorageShape *scale_shape, gert::StorageShape *offset_shape,
                        gert::StorageShape expanded_x_shape, gert::StorageShape expanded_row_idx_shape,
                        gert::StorageShape expert_tokens_count_or_cumsum_shape,
                        gert::StorageShape expert_tokens_before_capacity_shape, gert::StorageShape expanded_scale_shape,
                        int64_t activeNum, int64_t C, int64_t expert_num, int64_t dropPadMode, int64_t countFlag,
                        bool tokenFlag, int64_t quantMode, ge::DataType xDataType,
                        std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, ge::graphStatus result,
                        int64_t tilingKey)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 40}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MoeInitRoutingV3CompileInfo compile_info;

    std::string op_type("MoeInitRoutingV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    ge::DataType expandedXDataType = xDataType;
    if (quantMode != QUANT_MODE_NONE) {
        expandedXDataType = ge::DT_INT8;
    }

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(4, 5)
            .SetOpType(op_type)
            .IrInstanceNum({1, 1, 1, 1})
            .InputShapes({&x_shape, &expert_idx_shape, scale_shape, offset_shape})
            .OutputShapes({&expanded_x_shape, &expanded_row_idx_shape, &expert_tokens_count_or_cumsum_shape,
                           &expert_tokens_before_capacity_shape, &expanded_scale_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, expandedXDataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(activeNum)},
                        {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(C)},
                        {"expert_num", ge::AnyValue::CreateFrom<int64_t>(expert_num)},
                        {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                        {"expert_tokens_count_or_cumsum_flag", ge::AnyValue::CreateFrom<int64_t>(countFlag)},
                        {"expert_tokens_before_capacity_flag", ge::AnyValue::CreateFrom<bool>(tokenFlag)},
                        {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(quantMode)},
                        {"acive_expert_range", ge::AnyValue::CreateFrom<std::vector<int64_t>>(aciveExpertRange)},
                        {"row_idx_type", ge::AnyValue::CreateFrom<int64_t>(rowIdxType)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), result);
    if (result == ge::GRAPH_SUCCESS) {
        // todo check tiling result
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKey);
        auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
        std::cout << tiling_data_result << std::endl;
    }
}

void RunNormalCaseRegbase(int64_t N, int64_t H, int64_t K, int64_t C, int64_t dropPadMode, int64_t countFlag,
                          bool tokenFlag, int64_t quantMode, int64_t scaleFlag, ge::DataType xDataType,
                          std::vector<int64_t> aciveExpertRange, int64_t rowIdxType, ge::graphStatus result,
                          int64_t tilingKey)
{
    gert::StorageShape x_shape = {{N, H}, {N, H}};
    gert::StorageShape expert_idx_shape = {{N, K}, {N, K}};
    gert::StorageShape scale_shape_base = {{N}, {N}};
    gert::StorageShape *scale_shape = nullptr;
    gert::StorageShape *offset_shape = nullptr;
    int64_t activeNum = N * K;
    int64_t expert_num = EXPERT_NUM_REGBASE;

    int64_t E = aciveExpertRange[1] - aciveExpertRange[0];
    if (quantMode == QUANT_MODE_DYNAMIC) {
        scale_shape_base = {{E, H}, {E, H}};
        scale_shape = (scaleFlag == 0) ? nullptr : &scale_shape_base;
    } else if (quantMode == QUANT_MODE_STATIC) {
        scale_shape_base = {{E, 1}, {E, 1}};
        if (scaleFlag) {
            scale_shape_base = {{E, H}, {E, H}};
        }
        scale_shape = &scale_shape_base;
    } else {
        scale_shape_base = {{N}, {N}};
        scale_shape = (scaleFlag == 0) ? nullptr : &scale_shape_base;
    }

    gert::StorageShape expanded_x_shape = {{N * K, H}, {N * K, H}};
    gert::StorageShape expanded_row_idx_shape = {{N * K}, {N * K}};
    gert::StorageShape expert_tokens_count_or_cumsum_shape = {{E}, {E}};
    gert::StorageShape expert_tokens_before_capacity_shape = {{E}, {E}};
    gert::StorageShape expanded_scale_shape = {{N * K}, {N * K}};
    RunTestCaseRegbase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
                       expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, expanded_scale_shape,
                       activeNum, C, expert_num, dropPadMode, countFlag, tokenFlag, quantMode, xDataType,
                       aciveExpertRange, rowIdxType, result, tilingKey);
}

// 单核 + not quant + scale not None   1000000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_01)
{
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 1, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPPAD,
                         ge::GRAPH_SUCCESS, 1000000);
}

// 单核 + not quant + drop pad + scale None 1001000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_02)
{
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192}, ROW_IDX_TYPE_DROPLESS,
                         ge::GRAPH_SUCCESS, 1001000);
}

// 多核 + not quant + scale None 1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_03)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_INT8, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1100000);
}

// 多核 + not quant + drop pad + scale None 1101000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_04)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_NONE, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1101000);
}


// 单核 + dynamci quant + scale not None   1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_05)
{
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1020000);
}

// 单核 + dynamci quant + scale None 1020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_06)
{
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1020000);
}

// 单核 + dynamci quant + drop mode + scale not None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_07)
{
    RunNormalCaseRegbase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                         ge::GRAPH_SUCCESS, 1021000);
}

// 单核 + dynamci quant + drop mode + scale None  1021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_08)
{
    RunNormalCaseRegbase(8, 60, 32, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100}, ROW_IDX_TYPE_DROPLESS,
                         ge::GRAPH_SUCCESS, 1021000);
}

// 多核 + dynamci quant + scale not None  11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_09)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1120000);
}

// 多核 + dynamci quant + scale None 11020000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_10)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPPAD, ge::GRAPH_SUCCESS, 1120000);
}

// 多核 + dynamci quant + drop mode + scale not None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_11)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 1, ge::DT_FLOAT, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000);
}

// 多核 + dynamci quant + drop mode + scale None  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_12)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000);
}

// 多核 + dynamci quant + drop mode + scale None + bfloat16  11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_13)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_BF16, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000);
}

// 多核 + dynamci quant + drop mode + scale None + float16 11021000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_14)
{
    RunNormalCaseRegbase(160, 96, 1450, 0, 0, 1, true, QUANT_MODE_DYNAMIC, 0, ge::DT_FLOAT16, {0, 100},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_SUCCESS, 1121000);
}

// 单核 + static quant + drop mode + scale not None   1100000
TEST_F(MoeInitRoutingV3Tiling, moe_init_routing_v3_tiling_regbase_15)
{
    RunNormalCaseRegbase(1, 83, 27, 0, 0, 1, true, QUANT_MODE_STATIC, 1, ge::DT_FLOAT, {180, 192},
                         ROW_IDX_TYPE_DROPLESS, ge::GRAPH_FAILED, 1020000);
}