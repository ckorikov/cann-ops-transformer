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
 * \file test_interleave_rope.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_interleave_rope.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void interleave_rope(
    GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class interleave_rope_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "interleave_rope_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "interleave_rope_test TearDown\n" << endl;
    }
};

TEST_F(interleave_rope_test, test_case_0001)
{
    size_t xSize = 32 * 32 * 64 * sizeof(half);
    size_t ySize = 32 * 32 * 64 * sizeof(half);
    size_t cosSize = 32 * 1 * 64 * sizeof(half);
    size_t sinSize = 32 * 1 * 64 * sizeof(half);
    size_t tiling_data_size = sizeof(InterleaveRopeTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    InterleaveRopeTilingData* tilingDatafromBin = reinterpret_cast<InterleaveRopeTilingData*>(tiling);

    tilingDatafromBin->blockDim = 8;

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(interleave_rope, blockDim, x, cos, sin, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(interleave_rope_test, test_case_0002)
{
    size_t xSize = 32 * 32 * 64 * sizeof(half);
    size_t ySize = 32 * 32 * 64 * sizeof(half);
    size_t cosSize = 32 * 1 * 64 * sizeof(half);
    size_t sinSize = 32 * 1 * 64 * sizeof(half);
    size_t tiling_data_size = sizeof(InterleaveRopeTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    InterleaveRopeTilingData* tilingDatafromBin = reinterpret_cast<InterleaveRopeTilingData*>(tiling);

    tilingDatafromBin->blockDim = 8;

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(interleave_rope, blockDim, x, cos, sin, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(interleave_rope_test, test_case_0003)
{
    size_t xSize = 32 * 32 * 64 * sizeof(half);
    size_t ySize = 32 * 32 * 64 * sizeof(half);
    size_t cosSize = 32 * 1 * 64 * sizeof(half);
    size_t sinSize = 32 * 1 * 64 * sizeof(half);
    size_t tiling_data_size = sizeof(InterleaveRopeTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    InterleaveRopeTilingData* tilingDatafromBin = reinterpret_cast<InterleaveRopeTilingData*>(tiling);

    tilingDatafromBin->blockDim = 8;

    ICPU_SET_TILING_KEY(3000);
    ICPU_RUN_KF(interleave_rope, blockDim, x, cos, sin, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}