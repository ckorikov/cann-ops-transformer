/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_gating_top_k_softmax_tiling_base.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_BASE_H_

#include "moe_gating_top_k_softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
// #include "op_tiling_util.h"
// #include "runtime/runtime2_util.h"
// #include "register/op_compile_info_base.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxEKFullLoadTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, colAlign);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlignB16);
TILING_DATA_FIELD_DEF(uint32_t, kAlignB32);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);
TILING_DATA_FIELD_DEF(uint32_t, blockTail);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, formerSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, formerBlockTailSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, tailBlockTailSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, formerTopkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, formerBlockTailTopkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, tailBlockTailTopkTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxEKFullLoadTilingData);

BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxKFullLoadTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);   // 0 dim former block size
TILING_DATA_FIELD_DEF(uint32_t, blockTail);     // 0 dim tail block size
TILING_DATA_FIELD_DEF(uint32_t, ubLoop);        // 0 dim loop num
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);      // size of eash fromer loop
TILING_DATA_FIELD_DEF(uint32_t, ubFormerAlign); // align size of eash fromer loop
TILING_DATA_FIELD_DEF(uint32_t, ubTail);        // size of tail loop
TILING_DATA_FIELD_DEF(uint32_t, ubTailAlign);   // align size of tail loop
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, ubFormerSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, ubTailSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkFormerTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTailTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_6, MoeGatingTopKSoftmaxKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_7, MoeGatingTopKSoftmaxKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_8, MoeGatingTopKSoftmaxKFullLoadTilingData);

BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxPerfTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, colBytesAlign);
TILING_DATA_FIELD_DEF(uint32_t, colAlign);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);
TILING_DATA_FIELD_DEF(uint32_t, blockTail);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, topKValuesMask);
TILING_DATA_FIELD_DEF(uint32_t, topKIndicesMask);
TILING_DATA_FIELD_DEF(uint32_t, bufferElemSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_9, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_10, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_11, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_12, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_13, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_14, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_15, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_16, MoeGatingTopKSoftmaxPerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmax_17, MoeGatingTopKSoftmaxPerfTilingData);

class MoeGatingTopKSoftmaxBaseTiling : public TilingBaseClass
{
public:
    explicit MoeGatingTopKSoftmaxBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus CheckOutShape(const gert::Shape&, gert::Shape&);

protected:
    int coreNum;
    uint64_t ubSize;
    ge::DataType dtype;
    uint32_t row;
    uint32_t col;
    int64_t k;
};
} // namespace optiling

#endif