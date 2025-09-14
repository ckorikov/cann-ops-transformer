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
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "moe_compute_expert_tokens_tiling.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void moe_compute_expert_tokens(
    GM_ADDR sorted_experts, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

class moe_compute_expert_tokens_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "moe_compute_expert_tokens_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "moe_compute_expert_tokens_test TearDown\n" << std::endl;
    }
};

TEST_F(moe_compute_expert_tokens_test, test_case_int32_s)
{
    system(
        "cp -rf "
        "../../../../../../../ops/transformer/moe_compute_expert_tokens/tests/ut/op_kernel/moe_expert_data "
        "./");
    system("chmod -R 755 ./moe_expert_data/");
    system("cd ./moe_expert_data/ && python3 gen_data.py 381 1");
    system("cd ./moe_expert_data/ && python3 gen_tiling.py 'case0'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeComputeExpertTokensTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint32_t sortedExpertNum = 381;
    uint32_t outNum = 1;
    uint32_t blockDim = 48;

    size_t sortedExpertsByteSize = sortedExpertNum * sizeof(int32_t);
    size_t outByteSize = outNum * sizeof(int32_t);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * 32;

    uint8_t* in_sorted_experts = (uint8_t*)AscendC::GmAlloc(sortedExpertsByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(
        curPath + "/moe_expert_data/sorted_experts.bin", sortedExpertsByteSize, in_sorted_experts,
        sortedExpertsByteSize);
    ReadFile(curPath + "/moe_expert_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(moe_compute_expert_tokens, blockDim, in_sorted_experts, out, workSpace, tiling);

    WriteFile(curPath + "/moe_expert_data/output_actual.bin", out, outByteSize);
    AscendC::GmFree((void*)in_sorted_experts);
    AscendC::GmFree((void*)out);

    system("cd ./moe_expert_data/ && python3 compare_data.py 'int32'");
}

TEST_F(moe_compute_expert_tokens_test, test_case_int32m)
{
    system(
        "cp -rf "
        "../../../../../../../ops/transformer/moe_compute_expert_tokens/tests/ut/op_kernel/moe_expert_data "
        "./");
    system("chmod -R 755 ./moe_expert_data/");
    system("cd ./moe_expert_data/ && python3 gen_data.py  77 100");
    system("cd ./moe_expert_data/ && python3 gen_tiling.py 'case1'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeComputeExpertTokensTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint32_t sortedExpertNum = 77;
    uint32_t outNum = 100;
    uint32_t blockDim = 48;

    size_t sortedExpertsByteSize = sortedExpertNum * sizeof(int32_t);
    size_t outByteSize = outNum * sizeof(int32_t);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * 32;

    uint8_t* in_sorted_experts = (uint8_t*)AscendC::GmAlloc(sortedExpertsByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(
        curPath + "/moe_expert_data/sorted_experts.bin", sortedExpertsByteSize, in_sorted_experts,
        sortedExpertsByteSize);
    ReadFile(curPath + "/moe_expert_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(moe_compute_expert_tokens, blockDim, in_sorted_experts, out, workSpace, tiling);

    WriteFile(curPath + "/moe_expert_data/output_actual.bin", out, outByteSize);
    AscendC::GmFree((void*)in_sorted_experts);
    AscendC::GmFree((void*)out);

    system("cd ./moe_expert_data/ && python3 compare_data.py 'int32'");
}

TEST_F(moe_compute_expert_tokens_test, test_case_int32_l)
{
    system(
        "cp -rf "
        "../../../../../../../ops/transformer/moe_compute_expert_tokens/tests/ut/op_kernel/moe_expert_data "
        "./");
    system("chmod -R 755 ./moe_expert_data/");
    system("cd ./moe_expert_data/ && python3 gen_data.py  97601 8193");
    system("cd ./moe_expert_data/ && python3 gen_tiling.py 'case2'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(MoeComputeExpertTokensTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint32_t sortedExpertNum = 97601;
    uint32_t outNum = 8193;
    uint32_t blockDim = 48;

    size_t sortedExpertsByteSize = sortedExpertNum * sizeof(int32_t);
    size_t outByteSize = outNum * sizeof(int32_t);
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * 32;

    uint8_t* in_sorted_experts = (uint8_t*)AscendC::GmAlloc(sortedExpertsByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

    std::string curPath = ".";
    ReadFile(
        curPath + "/moe_expert_data/sorted_experts.bin", sortedExpertsByteSize, in_sorted_experts,
        sortedExpertsByteSize);
    ReadFile(curPath + "/moe_expert_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(1003);
    ICPU_RUN_KF(moe_compute_expert_tokens, blockDim, in_sorted_experts, out, workSpace, tiling);

    WriteFile(curPath + "/moe_expert_data/output_actual.bin", out, outByteSize);
    AscendC::GmFree((void*)in_sorted_experts);
    AscendC::GmFree((void*)out);

    system("cd ./moe_expert_data/ && python3 compare_data.py 'int32'");
}