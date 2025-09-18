#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "op_log.h"

#include "kernel_run_context_facker.h"

#include "fusion_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "mock/hcom_topo_info.h"
#include "test_cube_util.h"

class AlltoAllvGroupedMatMulTilingA5 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AlltoAllvGroupedMatMulTilingA5 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AlltoAllvGroupedMatMulTilingA5 TearDown" << std::endl;
    }
};

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_2) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{52428800, 7168}, {52428800, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{52428800, 1280}, {52428800, 1280}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&gmm_output_shape, nullptr, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&gmm_output_shape, nullptr, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    auto tilingKey = tiling_context->GetTilingKey();
    ASSERT_EQ(tilingKey, 1000000000000001000);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_shared_expert_h0) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 0}, {4096, 0}};
    gert::StorageShape gmmW_shape = {{4, 0, 1280}, {4, 0, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{1024, 7168}, {1024, 7168}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{1024, 8192}, {1024, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_shared_expert) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{1024, 7168}, {1024, 7168}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{1024, 8192}, {1024, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    auto tilingKey = tiling_context->GetTilingKey();
    ASSERT_EQ(tilingKey, 1000000000000001100);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_no_shared_expert_has_trans) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 1280, 7168}, {4, 1280, 7168}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{1024, 7168}, {1024, 7168}};
    gert::StorageShape mmw_shape = {{8192, 7168}, {8192, 7168}};
    gert::StorageShape mm_output_shape = {{1024, 8192}, {1024, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(true)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(true)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_shared_expert_trans) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 1280, 7168}, {4, 1280, 7168}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{1024, 7168}, {1024, 7168}};
    gert::StorageShape mmw_shape = {{8192, 7168}, {8192, 7168}};
    gert::StorageShape mm_output_shape = {{1024, 8192}, {1024, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(true)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(true)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    auto tilingKey = tiling_context->GetTilingKey();
    ASSERT_EQ(tilingKey, 1000000000000000111);
}


TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_unmatch_dataType) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&gmm_output_shape, nullptr, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 0);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_invalid_dataType) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{4096, 8192}, {4096, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 0);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_shared_expert_invalid_dim) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{4096, 7168}, {4096, 7163}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{4096, 8192}, {4096, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 0);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_shared_expert_invalid_dimNum) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{1, 4096, 7168}, {1, 4096, 7168}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{4096, 8192}, {4096, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 0);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_1_invalid_dimNum) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{1, 4096, 7168}, {1, 4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{4096, 8192}, {4096, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 100);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_not_support_permute_out) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmW_shape = {{4, 7168, 1280}, {4, 7168, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&gmm_output_shape, nullptr, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(true)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 0);
}

TEST_F(AlltoAllvGroupedMatMulTilingA5, allto_allv_grouped_mat_mul_test_tiling_float16_matmul_limits) {
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
        "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l12bt": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
        "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AlltoAllvGroupedMatMulCompileInfo {
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 3)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmx_shape = {{4096, 65550}, {4096, 65550}};
    gert::StorageShape gmmW_shape = {{4, 65550, 1280}, {4, 65550, 1280}};
    gert::StorageShape gmm_output_shape = {{4096, 1280}, {4096, 1280}};
    gert::StorageShape mmx_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape mmw_shape = {{7168, 8192}, {7168, 8192}};
    gert::StorageShape mm_output_shape = {{4096, 8192}, {4096, 8192}};

    std::vector<int64_t> sendCnt;
    std::vector<int64_t> recvCnt;
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j++) {
            recvCnt.push_back(4096 / 4 / 4);
            sendCnt.push_back(4096 / 4 / 4);
        }
    }

    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&gmmx_shape, &gmmW_shape, nullptr, nullptr, &mmx_shape, &mmw_shape})
                        .OutputShapes({&gmm_output_shape, &mm_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                                    {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)},
                                    {"send_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(sendCnt)},
                                    {"recv_counts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(recvCnt)},
                                    {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)},
                                    {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)

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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 2;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // auto tilingKey = tiling_context->GetTilingKey();
    // ASSERT_EQ(tilingKey, 100);
}
