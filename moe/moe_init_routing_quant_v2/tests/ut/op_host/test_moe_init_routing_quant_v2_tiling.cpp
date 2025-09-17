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
#include "transformer/moe_init_routing_quant_v2/op_host/moe_init_routing_quant_v2_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "fusion_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MoeInitRoutingQuantV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingQuantV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingQuantV2Tiling TearDown" << std::endl;
  }
};

static string TilingData2Str(const gert::TilingData* tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
    result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
    result += " ";
  }

  return result;
}

// optionalDtypePosi 0 1 2 分别代表设置DataType的位置
void RunTestCase(gert::StorageShape x_shape, gert::StorageShape expert_idx_shape, gert::StorageShape scale_shape,
                 gert::StorageShape offset_shape, gert::StorageShape expanded_x_shape,
                 gert::StorageShape expanded_row_idx_shape, gert::StorageShape expert_tokens_count_or_cumsum_shape,
                 gert::StorageShape expert_tokens_before_capacity_shape, gert::StorageShape dynamic_quant_scale_shape,
                 int64_t activeNum, int64_t C, int64_t E, int64_t dropPadMode, int64_t countFlag, bool tokenFlag,
                 int64_t quantMode, ge::DataType optionalDt, int64_t optionalDtypePosi, ge::graphStatus result, int64_t tilingKey) {
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
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::MoeInitRoutingQuantV2CompileInfo compile_info;

  std::string op_type("MoeInitRoutingQuantV2");
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
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  ge::DataType dtScale = optionalDtypePosi == 0 ? optionalDt : ge::DT_FLOAT;
  ge::DataType dtOffset = optionalDtypePosi == 1 ? optionalDt : ge::DT_FLOAT;
  ge::DataType dtDynamic = optionalDtypePosi == 2 ? optionalDt : ge::DT_FLOAT;

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(4, 5)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x_shape, &expert_idx_shape, &scale_shape, &offset_shape})
                    .OutputShapes({&expanded_x_shape, &expanded_row_idx_shape, &expert_tokens_count_or_cumsum_shape,
                                   &expert_tokens_before_capacity_shape, &dynamic_quant_scale_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, dtScale, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, dtOffset, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(4, dtDynamic, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"active_num", ge::AnyValue::CreateFrom<int64_t>(activeNum)},
                                {"expert_capacity", ge::AnyValue::CreateFrom<int64_t>(C)},
                                {"expert_num", ge::AnyValue::CreateFrom<int64_t>(E)},
                                {"drop_pad_mode", ge::AnyValue::CreateFrom<int64_t>(dropPadMode)},
                                {"expert_tokens_count_or_cumsum_flag", ge::AnyValue::CreateFrom<int64_t>(countFlag)},
                                {"expert_tokens_before_capacity_flag", ge::AnyValue::CreateFrom<bool>(tokenFlag)},
                                {"quant_mode", ge::AnyValue::CreateFrom<int64_t>(quantMode)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

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

void RunNormalCase(int64_t N, int64_t H, int64_t K, int64_t activeNum, int64_t C, int64_t E, int64_t dropPadMode,
                   int64_t countFlag, bool tokenFlag, int64_t quantMode, int64_t dqFlag, ge::DataType optionalOutputDt,
                   int64_t pos, ge::graphStatus result, int64_t tilingKey) {
  gert::StorageShape x_shape = {{N, H}, {N, H}};
  gert::StorageShape expert_idx_shape = {{N, K}, {N, K}};
  gert::StorageShape scale_shape = {{1}, {1}};
  gert::StorageShape offset_shape = {{1}, {1}};
  if (quantMode == 1) {
    if (dqFlag == 0) {
      scale_shape = {{E, H}, {E, H}};
    } else {
      scale_shape = {{1, H}, {1, H}};
    }
  }

  gert::StorageShape expanded_x_shape = {{E, C, H}, {E, C, H}};
  gert::StorageShape dynamin_quant_scale_shape = {{E * C}, {E * C}};
  if (dropPadMode == 0) {
    int64_t first_dim = N * K;
    if (activeNum > 0 && activeNum < first_dim) {
      first_dim = activeNum;
    }
    expanded_x_shape = {{first_dim, H}, {first_dim, H}};
    dynamin_quant_scale_shape = {{first_dim}, {first_dim}};
  }
  gert::StorageShape expanded_row_idx_shape = {{N * K}, {N * K}};
  gert::StorageShape expert_tokens_count_or_cumsum_shape = {{E}, {E}};
  gert::StorageShape expert_tokens_before_capacity_shape = {{E}, {E}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape,
              activeNum, C, E, dropPadMode, countFlag, tokenFlag, quantMode, optionalOutputDt, pos, result, tilingKey);
}

// 单核+静态quant+drop  10100
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_01) {
  RunNormalCase(8, 30, 6, 0, 6, 8, 1, 0, true, 0, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 10100);
}

// 单核+静态quant+dropless  10000
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_02) {
  RunNormalCase(80, 3000, 60, 0, 6, 8, 0, 1, false, 0, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 10000);
}

// 单核+动态quant+drop  11100
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_03) {
  RunNormalCase(8, 30, 6, 0, 6, 8, 1, 0, true, 1, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 11100);
}

// 单核+动态quant+dropless  11000
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_04) {
  RunNormalCase(80, 3000, 60, 0, 6, 8, 0, 1, false, 1, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 11000);
}

// 多核+静态quant+drop  10110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_05) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 1, 0, true, 0, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 10110);
}

// 多核+静态quant+drop + 切H  10110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_06) {
  RunNormalCase(320, 30000, 56, 0, 200, 32, 1, 0, true, 0, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 10110);
}

// 多核+静态quant+dropless  10010
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_07) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 0, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 10010);
}

// 多核+动态quant+drop  11110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_08) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 1, 0, true, 1, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 11110);
}

// 多核+动态quant+drop + 切H 11110
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_09) {
  RunNormalCase(320, 30000, 56, 0, 200, 32, 1, 0, true, 1, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 11110);
}

// 多核+动态quant+dropless  11010
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_10) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 1, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 11010);
}

// 性能模板+quant
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_11) {
  RunNormalCase(8, 30, 6, 32, 0, 8, 0, 1, false, 0, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 20000);
}
// 性能模板+dynamic quant
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_12) {
  RunNormalCase(8, 30, 6, 32, 0, 8, 0, 1, false, 1, 0, ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, 21000);
}

// failed
// quant mode != 0 or 1
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_13) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 100, 0, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 11010);
}

// scale_shape and offset_shape is wrong
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_14) {
  gert::StorageShape x_shape = {{320, 3000}, {320, 3000}};
  gert::StorageShape expert_idx_shape = {{320, 56}, {320, 56}};
  gert::StorageShape scale_shape = {{1, 2}, {1, 2}};
  gert::StorageShape offset_shape = {{1}, {1}};
  gert::StorageShape expanded_x_shape = {{17920, 3000}, {17920, 3000}};
  gert::StorageShape expanded_row_idx_shape = {{17920}, {17920}};
  gert::StorageShape expert_tokens_count_or_cumsum_shape = {{64}, {64}};
  gert::StorageShape expert_tokens_before_capacity_shape = {{64}, {64}};
  gert::StorageShape dynamin_quant_scale_shape = {{320, 1}, {320, 1}};

  // scale shape dim != 1 when quantMode = 0
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 0, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // scale shape first dim != 1 when quantMode = 0
  scale_shape = {{2}, {2}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 0, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // offset shape dim != 1 when quantMode = 0
  scale_shape = {{1}, {1}};
  offset_shape = {{1, 100}, {1, 100}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 0, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // offset shape first dim != 1 when quantMode = 0
  offset_shape = {{2}, {2}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 0, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // scale shape dim != 2 when quantMode = 1
  scale_shape = {{2}, {2}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 1, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // dynamic quant scale shape is not (E, H) or (1, H) when quantMode = 1
  scale_shape = {{100, 3000}, {100, 3000}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 1, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  scale_shape = {{1, 3456}, {1, 3456}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 1, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // dynamic quant scale shape dim != 1
  scale_shape = {{1, 3000}, {1, 3000}};
  dynamin_quant_scale_shape = {{2, 2}, {2, 2}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 1, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // dynamic quant scale shape is not (min(n*k, activeNum),) when drop pad mode = 0
  dynamin_quant_scale_shape = {{10000}, {10000}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 0, 1, false, 1, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);

  // dynamic quant scale shape is not (E * C,) when drop pad mode = 1
  dynamin_quant_scale_shape = {{10000}, {10000}};
  RunTestCase(x_shape, expert_idx_shape, scale_shape, offset_shape, expanded_x_shape, expanded_row_idx_shape,
              expert_tokens_count_or_cumsum_shape, expert_tokens_before_capacity_shape, dynamin_quant_scale_shape, 1000,
              0, 64, 1, 0, false, 1, ge::DT_FLOAT, 0, ge::GRAPH_FAILED, 0);
}

// scale dtype = int32
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_15) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 0, 0, ge::DT_INT32, 0, ge::GRAPH_FAILED, 0);
}

// scale dtype = int32 & quantMode = 1
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_16) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 1, 0, ge::DT_INT32, 0, ge::GRAPH_FAILED, 0);
}

// offset dtype = int32
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_17) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 0, 0, ge::DT_INT32, 1, ge::GRAPH_FAILED, 0);
}

// dynamic quant scale dtype = int32
TEST_F(MoeInitRoutingQuantV2Tiling, moe_init_routing_quant_v2_tiling_18) {
  RunNormalCase(320, 3000, 56, 0, 200, 32, 0, 1, false, 1, 0, ge::DT_INT32, 2, ge::GRAPH_FAILED, 0);
}