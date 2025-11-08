#include <iostream>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "rope_matrix_tiling.h"
using namespace matmul_tiling;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

uint8_t *GenerateTiling(RopeMatrixTiling *ropeTiling)
{
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

    int calSingleCoreM = B * H * M / usedCoreNum;
    int32_t baseM = 128;
    int32_t baseN = 128;

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
    tilingApi.SetSingleShape(calSingleCoreM, 128, 128);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << B << H << M << N << K << std::endl;
    }
    return GetTilingBuf(&tilingData);
}