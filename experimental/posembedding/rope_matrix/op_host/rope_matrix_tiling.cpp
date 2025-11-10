/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_matrix_tiling.h"

namespace RopeMatrix {
using namespace matmul_tiling;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint8_t *buf = nullptr;
    uint32_t tilingSize = tilingData->GetDataSize();
    if (tilingSize > 0) {
        buf = (uint8_t *)malloc(tilingSize);
        tilingData->SaveToBuffer(buf, tilingSize);
    }
    return buf;
}

uint8_t *GenerateTiling(RopeMatrixTiling *ropeTiling)
{
    constexpr uint32_t baseSize = 128;
    uint32_t usedCoreNum = ropeTiling->blockDim;
    uint32_t B = ropeTiling->b;
    uint32_t H = ropeTiling->n;
    uint32_t M = ropeTiling->s;
    uint32_t N = ropeTiling->d;
    uint32_t K = ropeTiling->d;

    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_BFLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_BFLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_BFLOAT16;

    bool isBias = false;

    uint32_t calSingleCoreM = B * H * M / usedCoreNum;
    uint32_t baseM = baseSize;
    uint32_t baseN = baseSize;

    optiling::TCubeTiling tilingData;
    const char *socVersion = "Ascend910B3";
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M * B * H, N, K); // 完成的MNK大小，单位为元素个数
    tilingApi.SetShape(M * B * H, N, K); // matmul计算形状的MNK，考虑脏数据
    tilingApi.SetSingleShape(calSingleCoreM, baseSize, baseSize);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    int64_t checkCode = -1;
    if (res == checkCode) {
        std::cout << "gen tiling failed" << B << H << M << N << K << std::endl;
    }
    return GetTilingBuf(&tilingData);
}

} // namespace RopeMatrix