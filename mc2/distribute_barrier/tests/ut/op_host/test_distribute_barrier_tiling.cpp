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
#include <map>
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "platform/platform_infos_def.h"

class DistributeBarrierTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DistributeBarrierTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "DistributeBarrierTiling TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling) {
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
            .KernelIONum(1, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_ref_shape = {{3, 4}, {3, 4}};
    gert::StorageShape x_ref_output_shape = {{3, 4}, {3, 4}};

    std::string group("group");
    int64_t world_size = 16;
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1})
                        .InputShapes({&x_ref_shape})
                        .OutputShapes({&x_ref_output_shape})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"world_size", ge::AnyValue::CreateFrom<int64_t>(world_size)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_world_size_1) {
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
            .KernelIONum(1, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_ref_shape = {{3, 4}, {3, 4}};
    gert::StorageShape x_ref_output_shape = {{3, 4}, {3, 4}};

    std::string group("group");
    int64_t world_size = 1;
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1})
                        .InputShapes({&x_ref_shape})
                        .OutputShapes({&x_ref_output_shape})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"world_size", ge::AnyValue::CreateFrom<int64_t>(world_size)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}


TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_world_size_385) {
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
            .KernelIONum(1, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_ref_shape = {{3, 4}, {3, 4}};
    gert::StorageShape x_ref_output_shape = {{3, 4}, {3, 4}};

    std::string group("group");
    int64_t world_size = 385;
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1})
                        .InputShapes({&x_ref_shape})
                        .OutputShapes({&x_ref_output_shape})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"world_size", ge::AnyValue::CreateFrom<int64_t>(world_size)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
