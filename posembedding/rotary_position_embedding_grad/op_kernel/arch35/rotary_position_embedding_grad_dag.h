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
 * \file rotary_position_embedding_grad_dag.h
 * \brief rotary_position_embedding_grad dag
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_DAG_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_DAG_H__

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace RotaryPositionEmbeddingGrad {
using namespace AscendC;
using namespace Ops::Base;
template <typename T, typename PromteT>
struct RotaryPositionEmbeddingGradDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using Mul0 = Bind<Vec::Mul<PromteT>, Cast0, Cast1>;
    using Reduce0 = Bind<Vec::ReduceOp<PromteT>, Mul0>;
    using Cast2 = Bind<Vec::Cast<T, PromteT, 1>, Reduce0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace RotaryPositionEmbeddingGrad

#endif