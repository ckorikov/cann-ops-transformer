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
#include "op_log.h"

#include "runtime2_util.h"
#include "kernel_run_context_facker.h"
#include "fusion_ops.h"
#include "array_ops.h"
#include "nn_other.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "transformer/grouped_matmul_finalize_routing/op_host/op_tiling/grouped_matmul_finalize_routing_base_tiling.h"
#include "transformer/grouped_matmul_finalize_routing/op_host/op_tiling/grouped_matmul_finalize_routing_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "matmul_v3_compile_info.h"
#include "op_tiling/cache_tiling.h"
#include "register/op_tiling_registry.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace optiling;

namespace optiling
{
extern void DisablePatternCache();
extern void EnablePatternCache();
}  // namespace optiling

class GroupedMatmulFinalizeRoutingTiling : public testing::Test
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

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestBF16ShouldFailed)
{

    std::string op_type("GroupedMatmulFinalizeRouting");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

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
    struct GroupedMatMulAllReduceCompileInfo {
    } compile_info;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(12016);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    int64_t dp = 4;
    int64_t batch = 1024;
    int64_t topK = 1;
    int64_t k = 1024;
    int64_t n = 8192;
    int64_t m = batch * topK;
    int64_t groupNum = topK;

    gert::StorageShape xShape = {{1280, 345}, {1280, 345}};
    gert::StorageShape wShape = {{1280, 567}, {1280, 567}};
    gert::StorageShape groupListShape = {{2}, {2}};
    gert::StorageShape yShape = {{690, 567}, {690, 567}};

    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(4, 1)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&xShape, &wShape, &groupListShape, &yShape})
                        .OutputShapes({&yShape})
                        .NodeAttrs({{"transpose_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"transpose_x", ge::AnyValue::CreateFrom<bool>(true)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // to check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestW4A8Case1)
{

    std::string op_type("GroupedMatmulFinalizeRouting");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

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
    struct GroupedMatMulAllReduceCompileInfo {
    } compile_info;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(12016);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    int64_t dp = 4;
    int64_t batch = 1024;
    int64_t topK = 1;
    int64_t k = 1024;
    int64_t n = 8192;
    int64_t m = batch * topK;
    int64_t groupNum = topK;

    gert::StorageShape xShape{{m, k}, {m, k}};
    gert::StorageShape wShape{{groupNum, k / 32, n / 16, 16, 32}, {groupNum, k / 32, n / 16, 16, 32}};
    gert::StorageShape groupListShape{{groupNum}, {groupNum}};
    gert::StorageShape yShape{{batch, n}, {batch, n}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xShape, &wShape, &groupListShape, &yShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs({
                          {"dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"sharedInputWeight", ge::AnyValue::CreateFrom<float>(1.0)},
                          {"sharedInputOffset", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"transposeX1", ge::AnyValue::CreateFrom<bool>(false)},
                          {"transposeX2", ge::AnyValue::CreateFrom<bool>(false)},
                          {"outputBS", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"groupListType", ge::AnyValue::CreateFrom<int64_t>(1)},
                      })
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .SetOpType(op_type)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // to check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 11000000000000000011UL);
}

TEST_F(GroupedMatmulFinalizeRoutingTiling, TestW8A8Case1)
{

    std::string op_type("GroupedMatmulFinalizeRouting");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

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
    struct GroupedMatMulAllReduceCompileInfo {
    } compile_info;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(12016);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    int64_t dp = 4;
    int64_t batch = 1024;
    int64_t topK = 1;
    int64_t k = 1024;
    int64_t n = 8192;
    int64_t m = batch * topK;
    int64_t groupNum = topK;

    gert::StorageShape xShape{{m, k}, {m, k}};
    gert::StorageShape wShape{{k, n}, {k, n}};
    gert::StorageShape groupListShape{{groupNum}, {groupNum}};
    gert::StorageShape yShape{{batch, n}, {batch, n}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xShape, &wShape, &groupListShape, &yShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs({
                          {"dtype", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"sharedInputWeight", ge::AnyValue::CreateFrom<float>(1.0)},
                          {"sharedInputOffset", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"transposeX1", ge::AnyValue::CreateFrom<bool>(false)},
                          {"transposeX2", ge::AnyValue::CreateFrom<bool>(false)},
                          {"outputBS", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"groupListType", ge::AnyValue::CreateFrom<int64_t>(1)},
                      })
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .SetOpType(op_type)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // to check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000000000000001UL);
}