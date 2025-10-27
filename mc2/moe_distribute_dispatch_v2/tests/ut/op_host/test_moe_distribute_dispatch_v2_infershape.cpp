/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#define private public
#include "platform/platform_info.h"

namespace MoeDistributeDispatchV2 {
class MoeDistributeDispatchV2Infershape : public testing::Test {
};

// infer shape with bias, success
TEST_F(MoeDistributeDispatchV2Infershape, inferShape0) 
{
    gert::StorageShape expand_x_shape = {{32, 7168}, {32, 7168}};
    gert::StorageShape expert_ids_shape = {{32, 8}, {32, 8}};

    gert::StorageShape expand_x_output_shape_uninit = {{10000, 10000}, {10000, 10000}};
    gert::StorageShape dynamic_scales_output_shape_uninit = {{10000}, {10000}};
    gert::StorageShape expand_idx_output_shape_uninit = {{10000}, {10000}};
    gert::StorageShape expert_token_nums_output_shape_uninit = {{10000}, {10000}};
    gert::StorageShape ep_recv_count_output_shape_uninit = {{10000}, {10000}};
    gert::StorageShape tp_recv_count_output_shape_uninit = {{10000}, {10000}};
    gert::StorageShape expand_scales_output_shape_uninit = {{10000}, {10000}};

    gert::InfershapeContextPara infershapeContextPara("MoeDistributeDispatchV2",
        {
            {expand_x_shape, ge::DT_INT32, ge::FORMAT_ND},
            {expert_ids_shape, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
        },
        {
            {expand_x_output_shape_uninit, ge::DT_INT32, ge::FORMAT_ND},
            {dynamic_scales_output_shape_uninit, ge::DT_INT32, ge::FORMAT_ND},
            {expand_idx_output_shape_uninit, ge::DT_INT32, ge::FORMAT_ND},
            {expert_token_nums_output_shape_uninit, ge::DT_INT32, ge::FORMAT_ND},
            {ep_recv_count_output_shape_uninit, ge::DT_INT32, ge::FORMAT_ND},
            {tp_recv_count_output_shape_uninit, ge::DT_INT32, ge::FORMAT_ND},
            {expand_scales_output_shape_uninit, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}
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
            {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{576, 7168}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}


TEST_F(MoeDistributeDispatchV2Infershape, inferDtype0) {
    ge::DataType expandXType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType expandXOutType = ge::DT_UNDEFINED; // 初始化的时候设置为未定义的类型
    ge::DataType dynamicScalesOutType = ge::DT_UNDEFINED;
    ge::DataType expandIdxOutType = ge::DT_UNDEFINED;
    ge::DataType expertTokenNumsOutType = ge::DT_UNDEFINED;
    ge::DataType epRecvCountOutType = ge::DT_UNDEFINED;
    ge::DataType tpRecvCountOutType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                      .IrInputNum(2)
                      .NodeIoNum(2, 6)
                      .IrInstanceNum({1, 1})
                      .NodeAttrs({
                        {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
                        {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                        {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
                        {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                        {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                        {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .NodeInputTd(0, expandXType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, expertIdsType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&expandXType, &expertIdsType})
                      .OutputDataTypes({&expandXOutType, &dynamicScalesOutType, &expandIdxOutType,
                                        &expertTokenNumsOutType, &epRecvCountOutType, &tpRecvCountOutType})
                      .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeDispatchV2")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}
}