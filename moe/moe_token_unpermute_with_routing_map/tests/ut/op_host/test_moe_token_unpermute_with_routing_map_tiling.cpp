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
#include "transformer/moe_token_unpermute_with_routing_map/op_host/moe_token_unpermute_with_routing_map_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MoeTokenUnpermuteWithRoutingMapTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenUnpermuteWithRoutingMapTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenUnpermuteWithRoutingMapTiling TearDown" << std::endl;
  }
};

// pad
TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_fp32_droppad) {
  dlog_setlevel(0, 0, 0);
  gert::StorageShape permuted_tokens_shape = {{4096*8, 7168}, {4096*8, 7168}};
  gert::StorageShape sorted_indices_shape = {{256*8}, {256*8}};
  gert::StorageShape routing_map = {{4096, 265}, {4096, 256}};
  gert::StorageShape prob_shape = {{4096, 8}, {4096, 8}};
  // output
  gert::StorageShape unpermuted_tokens_shape = {{8, 7168}, {8, 7168}};
  gert::StorageShape out_index_shape = {{256*8}, {256*8}};
  gert::StorageShape permute_token_id_shape = {{256*8}, {256*8}};
  gert::StorageShape permute_probs_shape = {{256*8}, {256*8}};
  vector<int64_t> restore_shape{4096, 7168};

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
  struct MoeTokenUnpermuteWithRoutingMapPadTilingData {};
  MoeTokenUnpermuteWithRoutingMapPadTilingData compile_info;

  std::string op_type("MoeTokenUnpermuteWithRoutingMap");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("MoeTokenUnpermuteWithRoutingMap")
                    .NodeIoNum(4, 4)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&permuted_tokens_shape, &sorted_indices_shape, &routing_map, &prob_shape})
                    .OutputShapes({&unpermuted_tokens_shape, &out_index_shape, &permute_token_id_shape, &permute_probs_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"drop_and_pad", ge::AnyValue::CreateFrom<bool>(true)},
                        {"restore_shape", ge::AnyValue::CreateFrom<vector<int64_t>>(restore_shape)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

  auto tiling_key = tiling_context->GetTilingKey();
  ASSERT_EQ(tiling_key, 1000);
  dlog_setlevel(0, 3, 0);
}

//notpad
TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_bf16) {
  dlog_setlevel(0, 0, 0);
  // uint64_t numTokens = 4096;
  // uint64_t hidden = 7168;
  // uint64_t numExperts = 4096;
  // uint64_t topkNum = 8;
  // uint64_t numOutTokens = 4096;
  // gert::StorageShape permutedTokensShape = {{numTokens*topkNum, hidden}, {numTokens*topkNum, hidden}};
  // gert::StorageShape sortedIndicesShape = {{numTokens*topkNum}, {numTokens*topkNum}};
  // gert::StorageShape routingMapShape = {{numTokens, numExperts}, {numTokens, numExperts}};
  // gert::StorageShape probsShape = {{numTokens, numExperts}, {numTokens, numExperts}};
  // // output
  // gert::StorageShape unpermutedTokensShape = {{numTokens, hidden}, {numTokens, hidden}};
  // gert::StorageShape outIndexShape = {{numTokens*topkNum}, {numTokens*topkNum}};
  // gert::StorageShape permuteTokenIdShape = {{numTokens*topkNum}, {numTokens*topkNum}};
  // gert::StorageShape permuteProbsShape = {{numTokens*topkNum}, {numTokens*topkNum}};
  // std::vector<int64_t> restoreShape({numTokens, hidden});

  gert::StorageShape permutedTokensShape = {{4096*8, 7168}, {4096*8, 7168}};
  gert::StorageShape sortedIndicesShape = {{256*8}, {256*8}};
  gert::StorageShape routingMapShape = {{4096, 265}, {4096, 256}};
  gert::StorageShape probsShape = {{4096, 8}, {4096, 8}};
  // output
  gert::StorageShape unpermutedTokensShape = {{8, 7168}, {8, 7168}};
  gert::StorageShape outIndexShape = {{256*8}, {256*8}};
  gert::StorageShape permuteTokenIdShape = {{256*8}, {256*8}};
  gert::StorageShape permuteProbsShape = {{256*8}, {256*8}};
  vector<int64_t> restoreShape{4096, 7168};

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
  struct MoeTokenUnpermuteWithRoutingMapPadTilingData {};
  MoeTokenUnpermuteWithRoutingMapPadTilingData compile_info;

  std::string op_type("MoeTokenUnpermuteWithRoutingMap");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  
  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 1024 * 1024);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("MoeTokenUnpermuteWithRoutingMap")
                    .NodeIoNum(4, 4)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&permutedTokensShape, &sortedIndicesShape, &routingMapShape, &probsShape})
                    .OutputShapes({&unpermutedTokensShape, &outIndexShape, &permuteTokenIdShape, &permuteProbsShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"drop_and_pad", ge::AnyValue::CreateFrom<bool>(false)},
                        {"restore_shape", ge::AnyValue::CreateFrom<std::vector<int64_t>>(restoreShape)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  dlog_setlevel(0, 3, 0);
}