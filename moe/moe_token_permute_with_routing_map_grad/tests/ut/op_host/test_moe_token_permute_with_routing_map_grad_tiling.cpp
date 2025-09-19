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
#include "transformer/moe_token_permute_with_routing_map_grad/op_host/moe_token_permute_with_routing_map_grad_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MoeTokenPermuteWithRoutingMapGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenPermuteWithRoutingMapGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenPermuteWithRoutingMapGradTiling TearDown" << std::endl;
  }
};

TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_bf16) {
  dlog_setlevel(0, 0, 0);
  gert::StorageShape permuted_output_d_shape = {{1024, 7168}, {1024, 7168}};
  gert::StorageShape prob_shape = {{1024}, {1024}};
  gert::StorageShape sorted_indices_shape = {{1024}, {1024}};
  gert::StorageShape routing_map = {{512, 512}, {512, 512}};
  // output
  gert::StorageShape tokens_grad_shape = {{512, 7168}, {512, 7168}};
  gert::StorageShape prob_grad_shape = {{512, 2}, {512, 2}};
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  struct MoeTokenPermuteWithEpGradCompileInfo {};
  MoeTokenPermuteWithEpGradCompileInfo compile_info;

  std::string op_type("MoeTokenPermuteWithRoutingMapGrad");
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

  // tilingParseFunc simulate
  auto kernel_holder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
          .Outputs({&compile_info})
          .Build();
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("MoeTokenPermuteWithRoutingMapGrad")
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&permuted_output_d_shape, &prob_shape, &sorted_indices_shape, &routing_map})
                    .OutputShapes({&tokens_grad_shape, &prob_grad_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"num_expert", ge::AnyValue::CreateFrom<int64_t>(512)},
                        {"tokens_num", ge::AnyValue::CreateFrom<int64_t>(512)},
                        {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0, 3, 0);
}


TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_fp32_droppad) {
  dlog_setlevel(0, 0, 0);
  gert::StorageShape permuted_output_d_shape = {{1024, 7168}, {1024, 7168}};
  gert::StorageShape prob_shape = {{1024}, {1024}};
  gert::StorageShape sorted_indices_shape = {{1024}, {1024}};
  gert::StorageShape routing_map = {{512, 512}, {512, 512}};
  // output
  gert::StorageShape tokens_grad_shape = {{512, 7168}, {512, 7168}};
  gert::StorageShape prob_grad_shape = {{512, 512}, {512, 512}};
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  struct MoeTokenPermuteWithEpGradCompileInfo {};
  MoeTokenPermuteWithEpGradCompileInfo compile_info;

  std::string op_type("MoeTokenPermuteWithRoutingMapGrad");
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

  // tilingParseFunc simulate
  auto kernel_holder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
          .Outputs({&compile_info})
          .Build();
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("MoeTokenPermuteWithRoutingMapGrad")
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&permuted_output_d_shape, &prob_shape, &sorted_indices_shape, &routing_map})
                    .OutputShapes({&tokens_grad_shape, &prob_grad_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"num_expert", ge::AnyValue::CreateFrom<int64_t>(512)},
                        {"tokens_num", ge::AnyValue::CreateFrom<int64_t>(512)},
                        {"padded_mode", ge::AnyValue::CreateFrom<bool>(true)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0, 3, 0);
}


TEST_F(MoeTokenPermuteWithRoutingMapGradTiling, test_tiling_invalid_TopK) {
  dlog_setlevel(0, 0, 0);
  gert::StorageShape permuted_output_d_shape = {{1024, 7168}, {1024, 7168}};
  gert::StorageShape prob_shape = {{1024}, {1024}};
  gert::StorageShape sorted_indices_shape = {{1024}, {1024}};
  gert::StorageShape routing_map = {{512, 512}, {512, 512}};
  // output
  gert::StorageShape tokens_grad_shape = {{512, 7168}, {512, 7168}};
  gert::StorageShape prob_grad_shape = {{512, 2}, {512, 2}};
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  struct MoeTokenPermuteWithEpGradCompileInfo {};
  MoeTokenPermuteWithEpGradCompileInfo compile_info;

  std::string op_type("MoeTokenPermuteWithRoutingMapGrad");
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

  // tilingParseFunc simulate
  auto kernel_holder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
          .Outputs({&compile_info})
          .Build();
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("MoeTokenPermuteWithRoutingMapGrad")
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&permuted_output_d_shape, &prob_shape, &sorted_indices_shape, &routing_map})
                    .OutputShapes({&tokens_grad_shape, &prob_grad_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"num_expert", ge::AnyValue::CreateFrom<int64_t>(512)},
                        {"tokens_num", ge::AnyValue::CreateFrom<int64_t>(1)},
                        {"padded_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
  dlog_setlevel(0, 3, 0);
}