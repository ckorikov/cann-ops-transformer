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
 * \file nsa_compress_grad.cpp
 * \brief nsa_compress_grad kernal
 */
#include "nsa_compress_grad.h"

using namespace AscendC;

using namespace NsaCompressGrad;

extern "C" __global__ __aicore__ void nsa_compress_grad(GM_ADDR outputGrad,
                                                        GM_ADDR input,
                                                        GM_ADDR weight,
                                                        GM_ADDR actSeqLenOptional,
                                                        GM_ADDR inputGrad,
                                                        GM_ADDR weightGrad,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
#if (ORIG_DTYPE_OUTPUTGRAD == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        NsaCompressGradND<half> op;
        op.Init(outputGrad, input, weight, actSeqLenOptional, inputGrad, weightGrad, userWs, &tilingData, &tPipe);
        op.Process();
        op.MoveResult();
        tPipe.Destroy();
    } 
#endif
#if (ORIG_DTYPE_OUTPUTGRAD == DT_BF16)
    if (TILING_KEY_IS(0)) {
        NsaCompressGradND<bfloat16_t> op;
        op.Init(outputGrad, input, weight, actSeqLenOptional, inputGrad, weightGrad, userWs, &tilingData, &tPipe);
        op.Process();
        op.MoveResult();
        tPipe.Destroy();
    }
#endif
}