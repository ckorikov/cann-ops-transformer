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
class SwinTransformerLnQkvQuantTiling : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "SwinTransformerLnQkvQuantTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "SwinTransformerLnQkvQuantTiling TearDown" << std::endl;
  }
};
TEST_F(SwinTransformerLnQkvQuantTiling, swin_transformer_ln_qkv_quant_tiling_1) {

  gert::StorageShape inputX_shape = {{1, 49, 96}, {1, 49, 96}};
  gert::StorageShape gamma_shape = {{96}, {96}};
  gert::StorageShape beta_shape = {{96}, {96}};
  gert::StorageShape weight_shape = {{288, 96}, {288, 96}};
  gert::StorageShape bias_shape = {{288}, {288}};
  gert::StorageShape quant_scale_shape = {{96}, {96}};
  gert::StorageShape quant_offset_shape = {{96}, {96}};
  gert::StorageShape dequant_scale_shape = {{288}, {288}};

  std::vector<int64_t> num_key_value_heads = {0};

  gert::StorageShape output_shapes = {{1,3,49,32}, {1,3,49,32}};

  string compile_info_string = R"({
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0","Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8}
                      })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}};
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);


  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct SwinTransformerLnQkvQuantCompileInfo {
      uint32_t coreNum = 8;
  };
  SwinTransformerLnQkvQuantCompileInfo compile_info;

  // std::map<std::string, std::string> platform_res;
  // platform_res["ai_core_cnt"] = std::to_string(24); // key
  // platform_info.SetPlatformRes("SoCInfo", platform_res); // label
  // platform_info.SetCoreNumByCoreType("AiCore"); // value

  std::string op_type("SwinTransformerLnQkvQuant");
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
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096 * 8);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);

  auto holder = gert::TilingContextFaker()
                    .SetOpType("SwinTransformerLnQkvQuant")
                    .NodeIoNum(8, 3)
                    .IrInstanceNum({1, 1,1,1,1,1,1,1})
                    .InputShapes({&inputX_shape, &gamma_shape, &beta_shape, &weight_shape, &bias_shape, &quant_scale_shape, &quant_offset_shape, &dequant_scale_shape})
                    .OutputShapes({&output_shapes, &output_shapes, &output_shapes})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(7, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"head_num", ge::AnyValue::CreateFrom<int64_t>(3)},
                      {"seq_length", ge::AnyValue::CreateFrom<int64_t>(32)},
                      {"epsilon", ge::AnyValue::CreateFrom<float>(0.00001)},
                      {"ori_height", ge::AnyValue::CreateFrom<int64_t>(7)},
                      {"ori_weight", ge::AnyValue::CreateFrom<int64_t>(7)},
                      {"h_win_size", ge::AnyValue::CreateFrom<int64_t>(7)},
                      {"w_win_size", ge::AnyValue::CreateFrom<int64_t>(7)},
                      {"weight_transpose", ge::AnyValue::CreateFrom<bool>(true)},
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
  //ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

//   auto block_dim = tiling_context->GetBlockDim();
//   ASSERT_EQ(block_dim, 48);

  // tiling_data not to check, because of exiting random num from tiling placehold
  // auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  // ASSERT_EQ(tiling_data_result, tiling_data);
}

TEST_F(SwinTransformerLnQkvQuantTiling, swin_transformer_ln_qkv_quant_tiling_subm_loop) {

  gert::StorageShape inputX_shape = {{3, 2912, 640}, {3, 2912, 640}};
  gert::StorageShape gamma_shape = {{640}, {640}};
  gert::StorageShape beta_shape = {{640}, {640}};
  gert::StorageShape weight_shape = {{1920, 640}, {1920, 640}};
  gert::StorageShape bias_shape = {{1920}, {1920}};
  gert::StorageShape quant_scale_shape = {{640}, {640}};
  gert::StorageShape quant_offset_shape = {{640}, {640}};
  gert::StorageShape dequant_scale_shape = {{1920}, {1920}};

  std::vector<int64_t> num_key_value_heads = {0};

  gert::StorageShape output_shapes = {{156,20,56,32}, {156,20,56,32}};

  string compile_info_string = R"({
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0","Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8}
                      })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}};
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);


  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct SwinTransformerLnQkvQuantCompileInfo {
      uint32_t coreNum = 8;
  };
  SwinTransformerLnQkvQuantCompileInfo compile_info;

  // std::map<std::string, std::string> platform_res;
  // platform_res["ai_core_cnt"] = std::to_string(24); // key
  // platform_info.SetPlatformRes("SoCInfo", platform_res); // label
  // platform_info.SetCoreNumByCoreType("AiCore"); // value

  std::string op_type("SwinTransformerLnQkvQuant");
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
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096 * 8);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);

  auto holder = gert::TilingContextFaker()
                    .SetOpType("SwinTransformerLnQkvQuant")
                    .NodeIoNum(8, 3)
                    .IrInstanceNum({1, 1,1,1,1,1,1,1})
                    .InputShapes({&inputX_shape, &gamma_shape, &beta_shape, &weight_shape, &bias_shape, &quant_scale_shape, &quant_offset_shape, &dequant_scale_shape})
                    .OutputShapes({&output_shapes, &output_shapes, &output_shapes})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(7, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"head_num", ge::AnyValue::CreateFrom<int64_t>(20)},
                      {"seq_length", ge::AnyValue::CreateFrom<int64_t>(32)},
                      {"epsilon", ge::AnyValue::CreateFrom<float>(0.00001)},
                      {"ori_height", ge::AnyValue::CreateFrom<int64_t>(32)},
                      {"ori_weight", ge::AnyValue::CreateFrom<int64_t>(91)},
                      {"h_win_size", ge::AnyValue::CreateFrom<int64_t>(8)},
                      {"w_win_size", ge::AnyValue::CreateFrom<int64_t>(7)},
                      {"weight_transpose", ge::AnyValue::CreateFrom<bool>(true)},
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
