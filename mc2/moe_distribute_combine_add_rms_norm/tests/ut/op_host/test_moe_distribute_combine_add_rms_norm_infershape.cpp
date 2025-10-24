/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <gtest/gtest.h>
 #include <iostream>
 #include "infer_shape_context_faker.h"
 #include "infer_datatype_context_faker.h"
 #include "infer_shape_case_executor.h"
 #include "base/registry/op_impl_space_registry_v2.h"

namespace MoeDistributeCombineAddRmsNormNameSpace{

 class MoeDistributeCombineAddRmsNorm : public testing::Test {
 protected:
     static void SetUpTestCase() {
         std::cout << "MoeDistributeCombineAddRmsNorm SetUp" << std::endl;
     }
 
     static void TearDownTestCase() {
         std::cout << "MoeDistributeCombineAddRmsNorm TearDown" << std::endl;
     }
 };
 
 // infer shape with bias, success
 TEST_F(MoeDistributeCombineAddRmsNorm, infer_shape_001)
 {
    gert::InfershapeContextPara infershapeContextPara("MoeDistributeCombineAddRmsNorm",
        {
            {{{1536, 7168}, {1536, 7168}}          , ge::DT_BF16, ge::FORMAT_ND},
            {{{192, 8}, {192, 8}}                  , ge::DT_INT32, ge::FORMAT_ND},
            {{{4608}, {4608}}                      , ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}                            , ge::DT_INT32, ge::FORMAT_ND},
            {{{192, 8}, {192, 8}}                  , ge::DT_FLOAT, ge::FORMAT_ND},
            {{{192, 1, 7168}, {192, 1, 7168}}      , ge::DT_BF16, ge::FORMAT_ND},
            {{{7168}, {7168}}                      , ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}                            , ge::DT_INT32, ge::FORMAT_ND},
            {{{192, 1, 7168}, {192, 1, 7168}}      , ge::DT_BF16, ge::FORMAT_ND}
            
        },
        {
            {{{192, 1, 7168}, {192, 1, 7168}}             ,ge::DT_BF16, ge::FORMAT_ND},
            {{{192, 1, 1}, {192, 1, 1}}                   ,ge::DT_FLOAT, ge::FORMAT_ND},
            {{{192, 1, 7168}, {192, 1, 7168}}             ,ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
            {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"shared_exper_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(22)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{192, 1, 7168}, {192, 1, 1}, {192, 1, 7168}};
    ExecuteTestCase(infershapeContextPara, ge::SUCCESS, expectOutputShape);
 }
 
 TEST_F(MoeDistributeCombineAddRmsNorm, infer_dtype_001)
 {
     ge::DataType expand_x_type = ge::DT_BF16;
     ge::DataType expert_ids_type = ge::DT_INT32;
     ge::DataType expand_idx_type = ge::DT_INT32;
     ge::DataType ep_send_counts_type = ge::DT_INT32;
     ge::DataType tp_send_counts_type = ge::DT_INT32;
     ge::DataType expert_scales_type = ge::DT_FLOAT;
     ge::DataType shared_expert_type = ge::DT_BF16;
     ge::DataType residual_type = ge::DT_BF16;
     ge::DataType gamma_type = ge::DT_BF16;
     ge::DataType dynamic_scale_type = ge::DT_FLOAT;
 
     std::string opType("MoeDistributeCombineAddRmsNorm");
 
     auto holder = gert::InferDataTypeContextFaker()
         .IrInputNum(2)
         .NodeIoNum(9, 3)
         .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
         .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeInputTd(8, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
         .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
         .InputDataTypes({&expand_x_type, &expert_ids_type, &expand_idx_type,
                          &ep_send_counts_type, &expert_scales_type,
                          &residual_type, &gamma_type, &tp_send_counts_type, &shared_expert_type})
         .OutputDataTypes({&residual_type, &dynamic_scale_type, &shared_expert_type})
         .Build();
 
     auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
     auto inferDtypeFunc = spaceRegistry->GetOpImpl(opType.c_str())->infer_datatype;
     ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
 
     EXPECT_EQ(holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
 }
} // MoeDistributeCombineAddRmsNormNameSpace