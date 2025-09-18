#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "op_log.h"

#include "kernel_run_context_facker.h"

#include "fusion_ops.h"
#include "array_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "test_cube_util.h"
using namespace std;
class SwinTransformerLnQKVTiling : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "SwinTransformerLnQKVTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "SwinTransformerLnQKVTiling TearDown" << std::endl;
  }
};
TEST_F(SwinTransformerLnQKVTiling, swin_transformer_ln_qkv_tiling_1) {

  gert::StorageShape inputX_shape = {{8, 65536, 128}, {1, 1000, 5120}};
  gert::StorageShape gamma_shape = {{128}, {1, 1000, 5120}};
  gert::StorageShape beta_shape = {{128}, {1, 1000, 5120}};
  gert::StorageShape weight_shape = {{128, 384}, {1, 1000, 5120}};
  gert::StorageShape bias_shape = {{384}, {1, 1000, 5120}};
  int64_t num_heads = 4;
  float epsilon = 1.0f;
  int64_t pre_tokens = 0;
  int64_t next_tokens = 0;
  std::vector<int64_t> num_key_value_heads = {0};

  gert::StorageShape output_shapes = {{8192, 4, 64, 32}, {8192, 4, 64, 32}};

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
  struct SwinTransformerLnQKVCompileInfo {
      uint32_t coreNum = 0;
  };
  SwinTransformerLnQKVCompileInfo compile_info;

  // std::map<std::string, std::string> platform_res;
  // platform_res["ai_core_cnt"] = std::to_string(24); // key
  // platform_info.SetPlatformRes("SoCInfo", platform_res); // label
  // platform_info.SetCoreNumByCoreType("AiCore"); // value

  std::string op_type("SwinTransformerLnQKV");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 3)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&inputX_shape, &gamma_shape, &beta_shape, &weight_shape, &bias_shape})
                    .OutputShapes({&output_shapes, &output_shapes, &output_shapes})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"epsilon", ge::AnyValue::CreateFrom<float>(1.0)},
                      {"head_dim", ge::AnyValue::CreateFrom<int64_t>(pre_tokens)},
                      {"head_num", ge::AnyValue::CreateFrom<int64_t>(next_tokens)},
                      {"seq_length", ge::AnyValue::CreateFrom<int64_t>(num_heads)},
                      {"shifts", ge::AnyValue::CreateFrom<std::vector<int64_t>>(num_key_value_heads)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

//   auto block_dim = tiling_context->GetBlockDim();
//   ASSERT_EQ(block_dim, 48);

  // tiling_data not to check, because of exiting random num from tiling placehold
  // auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  // ASSERT_EQ(tiling_data_result, tiling_data);
}