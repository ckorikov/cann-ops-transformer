/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_infershape.cc
 * \brief
 */
#include "op_util.h"
#include "op_const.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops
{
static const size_t INDEX_IN_GMM_X = 0;
static const size_t INDEX_IN_GMM_WEIGHT = 1;
static const size_t INDEX_IN_MM_X = 4;
static const size_t INDEX_IN_MM_WEIGHT = 5;
static const size_t INDEX_OUT_GMM_Y = 0;
static const size_t INDEX_OUT_MM_Y = 1;
static const size_t INDEX_PERMUTE_OUT = 2;
static const size_t INDEX_ATTR_EP_WORLD_SIZE = 1;
static const size_t INDEX_ATTR_SEND_COUNTS = 2;
static const size_t INDEX_ATTR_RECV_COUNTS = 3;
static const size_t INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
static const size_t INDEX_ATTR_TRANS_MM_WEIGHT_INDEX = 5;
static const size_t INDEX_ATTR_PERMUTE_OUT_FLAG_INDEX = 6;

static constexpr size_t DIM_NUM_0 = 0;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;

static constexpr int64_t FIRST_ELE_SIZE = -1;

static graphStatus CheckDims(const gert::InferShapeContext* context, const gert::Shape* gmmXShape,
                             const gert::Shape* gmmWeightShape, bool transGmmWeight)
{
    auto result = ge::GRAPH_SUCCESS;

    auto gmmXDimNum = gmmXShape->GetDimNum();
    auto gmmWeightDimNum = gmmWeightShape->GetDimNum();
    if (gmmXDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only gmmX with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (gmmWeightDimNum != DIM_NUM_3) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only gmmWeight with dim 3 is supported.");
        result = ge::GRAPH_FAILED;
    }

    auto k1 = gmmXShape->GetDim(DIM_1);
    auto k2 = gmmWeightShape->GetDim(DIM_1);
    if (transGmmWeight) {
        k2 = gmmWeightShape->GetDim(DIM_2);
    }
    if (k1 != k2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "Dim of gmmX and dim of gmmWeight do not match for MatMul");
        result = ge::GRAPH_FAILED;
    }

    return result;
}

static graphStatus CheckDimsOptional(const gert::InferShapeContext* context, const gert::Shape* mmXShape,
                                     const gert::Shape* mmWeightShape, bool transMmWeight)
{
    auto result = ge::GRAPH_SUCCESS;

    auto xDimNum = mmXShape->GetDimNum();
    auto mmWeightDimNum = mmWeightShape->GetDimNum();
    if (xDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only x with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (mmWeightDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only mmWeight with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }

    auto k1 = mmXShape->GetDim(DIM_1);
    auto k2 = mmWeightShape->GetDim(DIM_0);
    if (transMmWeight) {
        k2 = mmWeightShape->GetDim(DIM_1);
    }
    if (k1 != k2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "Dim of x and dim of mmWeight do not match for MatMul");
        result = ge::GRAPH_FAILED;
    }

    return result;
}

static ge::graphStatus InferShapeAlltoAllvGroupedMatMul(gert::InferShapeContext* context)
{
    auto* gmmXShape = context->GetInputShape(INDEX_IN_GMM_X);
    auto* gmmWeightShape = context->GetInputShape(INDEX_IN_GMM_WEIGHT);
    auto* mmXShape = context->GetOptionalInputShape(INDEX_IN_MM_X);
    auto* mmWeightShape = context->GetOptionalInputShape(INDEX_IN_MM_WEIGHT);
    auto* gmmYShape = context->GetOutputShape(INDEX_OUT_GMM_Y);
    auto* mmYShape = context->GetOutputShape(INDEX_OUT_MM_Y);
    auto* permuteOutShape = context->GetOutputShape(INDEX_PERMUTE_OUT);

    auto* attrs = context->GetAttrs();
    auto* epWorldSizePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_EP_WORLD_SIZE);
    auto* recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_RECV_COUNTS);
    auto* sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_SEND_COUNTS);
    auto* transGmmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto* transMmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_MM_WEIGHT_INDEX);
    auto* permuteOutFlagPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_PERMUTE_OUT_FLAG_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmXShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmWeightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmYShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transGmmWeightPtr);
    OP_CHECK(CheckDims(context, gmmXShape, gmmWeightShape, *transGmmWeightPtr) != ge::GRAPH_SUCCESS,
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckDims failed."), return ge::GRAPH_FAILED);

    int64_t E = gmmWeightShape->GetDim(DIM_0);
    int64_t A = 0;
    int64_t H = gmmXShape->GetDim(DIM_1);
    int64_t N1 = *transGmmWeightPtr ? gmmWeightShape->GetDim(DIM_1) : gmmWeightShape->GetDim(DIM_2);
    
    gmmYShape->SetDimNum(DIM_NUM_2);
    gmmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
    gmmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
    if (E != FIRST_ELE_SIZE) {
        int64_t arraySize = E * *epWorldSizePtr;
        OP_CHECK(recvCountsPtr->GetSize() < static_cast<size_t>(arraySize),
                 VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                     "recvCounts size should not be smaller than E * epWorldSize."),
                 return ge::GRAPH_FAILED);
        OP_CHECK(sendCountsPtr->GetSize() < static_cast<size_t>(arraySize),
                 VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                     "sendCounts size should not be smaller than E * epWorldSize."),
                 return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < arraySize; i++) {
            A += static_cast<const int64_t*>(recvCountsPtr->GetData())[i];
        }
        gmmYShape->SetDim(DIM_0, A);
        gmmYShape->SetDim(DIM_1, N1);
    }

    OP_CHECK(mmYShape == nullptr,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "the shape of output mm_y is nullptr."),
            return ge::GRAPH_FAILED);
    mmYShape->SetDimNum(DIM_NUM_0);
    if (mmXShape != nullptr && mmWeightShape != nullptr && mmYShape != nullptr && transMmWeightPtr != nullptr) {
        OP_CHECK(CheckDimsOptional(context, mmXShape, mmWeightShape, *transMmWeightPtr) != ge::GRAPH_SUCCESS,
                 VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckDimsOptional failed."),
                 return ge::GRAPH_FAILED);
        int64_t BS = mmXShape->GetDim(DIM_0);
        mmYShape->SetDimNum(DIM_NUM_2);
        mmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
        mmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        if (BS != FIRST_ELE_SIZE) {
            int64_t N2 = *transMmWeightPtr ? mmWeightShape->GetDim(DIM_0) : mmWeightShape->GetDim(DIM_1);
            mmYShape->SetDimNum(DIM_NUM_2);
            mmYShape->SetDim(DIM_0, BS);
            mmYShape->SetDim(DIM_1, N2);
        }
    }

    OP_CHECK(permuteOutShape == nullptr,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "the shape of output permute_out is nullptr."),
            return ge::GRAPH_FAILED);
    permuteOutShape->SetDimNum(DIM_NUM_0);
    if (permuteOutShape != nullptr && permuteOutFlagPtr != nullptr && *permuteOutFlagPtr == true) {
        permuteOutShape->SetDimNum(DIM_NUM_2);
        permuteOutShape->SetDim(DIM_0, FIRST_ELE_SIZE);
        permuteOutShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        if (E != FIRST_ELE_SIZE) {
            permuteOutShape->SetDim(DIM_0, A);
            permuteOutShape->SetDim(DIM_1, H);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllvGroupedMatMul(gert::InferDataTypeContext* context)
{
    auto dType = context->GetInputDataType(INDEX_IN_GMM_X);
    context->SetOutputDataType(INDEX_OUT_GMM_Y, dType);
    context->SetOutputDataType(INDEX_OUT_MM_Y, dType);
    context->SetOutputDataType(INDEX_PERMUTE_OUT, dType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllvGroupedMatMul)
    .InferShape(InferShapeAlltoAllvGroupedMatMul)
    .InferDataType(InferDataTypeAlltoAllvGroupedMatMul);
}  // namespace ops
