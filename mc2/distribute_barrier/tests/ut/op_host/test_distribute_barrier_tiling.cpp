/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class DistributeBarrierTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DistributeBarrierTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "DistributeBarrierTiling TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    std::string expectTilingData = "8 16 20 196352 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 42;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_world_size_1)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_world_size_385)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(385)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_elastic_info) {
    std::string op_type("DistributeBarrier");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 1024,
                                            "load3d_constraints": "0",
                                            "Intrinsic_fix_pipe_l0c2out": true,
                                            "Intrinsic_data_move_l12ub": false,
                                            "Intrinsic_data_move_l0c2ub": false,
                                            "Intrinsic_data_move_out2l1_nd2nz": true,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 20
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DistributeBarrierCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(3, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_ref_shape = {{3, 4}, {3, 4}};
    gert::StorageShape elastic_info_shape = {{36}, {36}};
    gert::StorageShape x_ref_output_shape = {{3, 4}, {3, 4}};

    std::string group("group");
    int64_t world_size = 16;
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(3, 1)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({&x_ref_shape, nullptr, &elastic_info_shape})
                        .OutputShapes({&x_ref_output_shape})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"world_size", ge::AnyValue::CreateFrom<int64_t>(world_size)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend910_93"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_time_out) {
    std::string op_type("DistributeBarrier");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 1024,
                                            "load3d_constraints": "0",
                                            "Intrinsic_fix_pipe_l0c2out": true,
                                            "Intrinsic_data_move_l12ub": false,
                                            "Intrinsic_data_move_l0c2ub": false,
                                            "Intrinsic_data_move_out2l1_nd2nz": true,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 20
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DistributeBarrierCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(3, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_ref_shape = {{3, 4}, {3, 4}};
    gert::StorageShape time_out_shape = {{1}, {1}};
    gert::StorageShape x_ref_output_shape = {{3, 4}, {3, 4}};

    std::string group("group");
    int64_t world_size = 16;
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(3, 1)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({&x_ref_shape, &time_out_shape, nullptr})
                        .OutputShapes({&x_ref_output_shape})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"world_size", ge::AnyValue::CreateFrom<int64_t>(world_size)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend910_93"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}