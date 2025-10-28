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

class MatmulAllReduceInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceInfershape TearDown" << std::endl;
    }
};

TEST_F(MatmulAllReduceInfershape, infer_shape_for_2dim) {
    gert::StorageShape x1_shape = {{32, 64}, {4, 2, 16, 16}};
    gert::StorageShape x2_shape = {{64, 128}, {4, 2, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_3dim) {
    gert::StorageShape x1_shape = {{4, 8, 64}, {4, 2, 16, 16}};
    gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_invalid_k) {
    gert::StorageShape x1_shape = {{32, 8}, {4, 2, 16, 16}};
    gert::StorageShape x2_shape = {{64, 128}, {4, 2, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_invalid_zero_k) {
    gert::StorageShape x1_shape = {{32, 0}, {4, 2, 16, 16}};
    gert::StorageShape x2_shape = {{0, 128}, {4, 2, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_dtype) {
    ge::DataType x1 = ge::DT_FLOAT16;
    ge::DataType x2 = ge::DT_FLOAT16;
    ge::DataType y = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                    .IrInputNum(2)
                    .NodeIoNum(3, 1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .InputDataTypes({&x1, &x2})
                    .OutputDataTypes({&y})
                    .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MatmulAllReduce")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), y);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_for_3dim_quant_v4)
{
    gert::StorageShape x1_shape = {{4, 8, 64}, {4, 2, 16, 16}};
    gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape output_shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_shape_add_rms_norm) {
    gert::StorageShape x1_shape = {{4, 8, 64}, {4, 2, 16, 16}};
    gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape residual_shape = {{4, 8, 128}, {4, 2, 16, 16}};
    gert::StorageShape output_shape_1 = {{}, {}};
    gert::StorageShape output_shape_2 = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara("MatmulAllReduce",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_INT32, ge::FORMAT_ND},
            {bias_shape, ge::DT_INT32, ge::FORMAT_ND},
            {residual_shape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {output_shape_1, ge::DT_FLOAT16, ge::FORMAT_ND},
            {output_shape_2, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
    );

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(MatmulAllReduceInfershape, infer_dtype_add_rms_norm) {
    ge::DataType x1 = ge::DT_FLOAT16;
    ge::DataType x2 = ge::DT_FLOAT16;
    ge::DataType bias = ge::DT_FLOAT16;
    ge::DataType residual = ge::DT_FLOAT16;
    ge::DataType y1 = ge::DT_FLOAT16;
    ge::DataType y2 = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                    .IrInputNum(4)
                    .NodeIoNum(4, 2)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .InputDataTypes({&x1, &x2, &bias, &residual})
                    .OutputDataTypes({&y1, &y2})
                    .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MatmulAllReduce")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), y1);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), y2);
}