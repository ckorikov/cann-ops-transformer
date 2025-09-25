/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_unpermute_with_routing_map_grad.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_moe_token_unpermute_with_routing_map_grad.h"
#include "../../../../../built-in/tests//ut//fast_op_test/data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map_grad(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR out_index, GM_ADDR permute_token_id, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace,
    GM_ADDR tiling);

class moe_token_unpermute_with_routing_map_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "moe_token_unpermute_with_routing_map_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "moe_token_unpermute_with_routing_map_grad_test TearDown\n" << endl;
    }
};

TEST_F(moe_token_unpermute_with_routing_map_grad_test, test_bf16_prob_not_none)
{
    size_t unpermutedTokensGradByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t outIndexByteSize = 30 * sizeof(int32_t);
    size_t permuteTokenIdByteSize = 30 * sizeof(int32_t);
    size_t routingMapByteSize = 30 * 8;
    size_t permutedTokensByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t probsByteSize = 30 * 8 * sizeof(bfloat16_t);

    // output
    size_t tokensGradOutByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t probsGradOutOptionalByteSize = 30 * 8 * sizeof(bfloat16_t);

    size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapGradTilingData);

    uint8_t* unpermutedTokensGrad = (uint8_t*)AscendC::GmAlloc(unpermutedTokensGradByteSize);
    uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
    uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
    uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
    uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);

    uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
    uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteWithRoutingMapGradTilingData* tilingDatafromBin =
        reinterpret_cast<MoeTokenUnpermuteWithRoutingMapGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        moe_token_unpermute_with_routing_map_grad, blockDim, unpermutedTokensGrad, outIndex, permuteTokenId, routingMap,
        permutedTokens, probs, tokensGradOut, probsGradOutOptional, workspace, tiling);

    AscendC::GmFree(unpermutedTokensGrad);
    AscendC::GmFree(outIndex);
    AscendC::GmFree(permuteTokenId);
    AscendC::GmFree(routingMap);
    AscendC::GmFree(permutedTokens);
    AscendC::GmFree(probs);
    AscendC::GmFree(tokensGradOut);
    AscendC::GmFree(probsGradOutOptional);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_grad_test, test_bf16_prob_none_pad_false)
{
    size_t unpermutedTokensGradByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t outIndexByteSize = 30 * sizeof(int32_t);
    size_t permuteTokenIdByteSize = 30 * sizeof(int32_t);
    size_t routingMapByteSize = 30 * 8;
    size_t permutedTokensByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t probsByteSize = 30 * 8 * sizeof(bfloat16_t);

    // output
    size_t tokensGradOutByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t probsGradOutOptionalByteSize = 30 * 8 * sizeof(bfloat16_t);

    size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapGradTilingData);

    uint8_t* unpermutedTokensGrad = (uint8_t*)AscendC::GmAlloc(unpermutedTokensGradByteSize);
    uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
    uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
    uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
    uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);

    uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
    uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteWithRoutingMapGradTilingData* tilingDatafromBin =
        reinterpret_cast<MoeTokenUnpermuteWithRoutingMapGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        moe_token_unpermute_with_routing_map_grad, blockDim, unpermutedTokensGrad, outIndex, permuteTokenId, routingMap,
        permutedTokens, probs, tokensGradOut, probsGradOutOptional, workspace, tiling);

    AscendC::GmFree(unpermutedTokensGrad);
    AscendC::GmFree(outIndex);
    AscendC::GmFree(permuteTokenId);
    AscendC::GmFree(routingMap);
    AscendC::GmFree(permutedTokens);
    AscendC::GmFree(probs);
    AscendC::GmFree(tokensGradOut);
    AscendC::GmFree(probsGradOutOptional);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_grad_test, test_bf16_prob_none_pad_true)
{
    size_t unpermutedTokensGradByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t outIndexByteSize = 30 * sizeof(int32_t);
    size_t permuteTokenIdByteSize = 30 * sizeof(int32_t);
    size_t routingMapByteSize = 30 * 8;
    size_t permutedTokensByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t probsByteSize = 30 * 8 * sizeof(bfloat16_t);

    // output
    size_t tokensGradOutByteSize = 30 * 8 * sizeof(bfloat16_t);
    size_t probsGradOutOptionalByteSize = 30 * 8 * sizeof(bfloat16_t);

    size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapGradTilingData);

    uint8_t* unpermutedTokensGrad = (uint8_t*)AscendC::GmAlloc(unpermutedTokensGradByteSize);
    uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
    uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
    uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
    uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
    uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);

    uint8_t* tokensGradOut = (uint8_t*)AscendC::GmAlloc(tokensGradOutByteSize);
    uint8_t* probsGradOutOptional = (uint8_t*)AscendC::GmAlloc(probsGradOutOptionalByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    MoeTokenUnpermuteWithRoutingMapGradTilingData* tilingDatafromBin =
        reinterpret_cast<MoeTokenUnpermuteWithRoutingMapGradTilingData*>(tiling);

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        moe_token_unpermute_with_routing_map_grad, blockDim, unpermutedTokensGrad, outIndex, permuteTokenId, routingMap,
        permutedTokens, probs, tokensGradOut, probsGradOutOptional, workspace, tiling);

    AscendC::GmFree(unpermutedTokensGrad);
    AscendC::GmFree(outIndex);
    AscendC::GmFree(permuteTokenId);
    AscendC::GmFree(routingMap);
    AscendC::GmFree(permutedTokens);
    AscendC::GmFree(probs);
    AscendC::GmFree(tokensGradOut);
    AscendC::GmFree(probsGradOutOptional);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}