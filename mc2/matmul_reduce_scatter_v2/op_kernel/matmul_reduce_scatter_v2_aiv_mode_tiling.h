/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_aiv_mode_tiling.h
 * \brief
 */
#ifndef __MATMUL_REDUCE_SCATTER_V2_AIV_MODE_TILING_H__
#define __MATMUL_REDUCE_SCATTER_V2_AIV_MODE_TILING_H__
#include <vector>
#include <map>
namespace matmulReduceScatterV2_aivmode_tiling{

enum class DequantType : int {
    DEQUANT_TYPE_UNDEFINED = -1,
    PER_CHANNEL = 0,
    PER_TOKEN = 1,
    DEQUANT_TYPE_MAX = 2,
};

struct MatmulReduceScatterV2AivModeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t aivNum;
    uint32_t totalUbSize;
    bool isTransposeA;
    bool isTransposeB;
    uint64_t aAlignSize;
    uint64_t bAlignSize;
    bool hasAAlign;
    bool hasBAlign;
    bool quantFlag;
    bool is910C;
    bool isX2ScaleTypeInt64; /* 当前x2Scale存在入参类型为int64的场景，需要tilingKey的方式传入kernel */
    DequantType dequant_type;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t mLoop = -1;
    int32_t kLoop = -1;
    int32_t nLoop = -1;
    int32_t swizzlCount = -1;
    int32_t swizzlDirect = -1;
    int32_t pValue = -1;
    int32_t ubMoveNum = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t lenPerLoop = -1;
};

struct MatmulReduceScatterV2AivModeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MatmulReduceScatterV2AivModeInfo matmulReduceScatterV2AivModeInfo;
    CoCTiling cocTiling;
};

struct MatmulReduceScatterV2AivModeTilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};
    explicit MatmulReduceScatterV2AivModeTilingValue(int32_t v = -1, 
        std::map<int, std::vector<std::vector<int>>> m = {}) 
        : value(v), conditionMap(std::move(m)) {}
};

constexpr int32_t REDUCESCATTER_M0_DEFAULT = 128;
constexpr int32_t REDUCESCATTER_UBMOVENUM_DEFAULT = 20;
constexpr int32_t REDUCESCATTER_COMMDATASPLIT_DEFAULT = 16;
constexpr int32_t REDUCESCATTER_PVALUE_DEFAULT = 12;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t SWIZZLE_DIRECT_ONE = 1;
constexpr int32_t SWIZZLE_COUNT_FOUR = 4;
constexpr int32_t COMM_DATA_DIRECT = 0;
constexpr int32_t COMMNPUSPLIT_ONE = 1;
constexpr int32_t HALF_KBYTE = 512;
constexpr int32_t DEFAULT_ROW = 128;
constexpr int32_t DEFAULT_COL = 256;

constexpr int32_t REDUCESCATTER_FOUR_RANK_PVALUE = 14;
constexpr int32_t COMMDATASPLIT_SIXTEEN = 16;
constexpr int32_t REDUCESCATTER_FOUR_RANK_UBMOVENUM = 8;
constexpr int32_t DEFAULT_SWIZZLE_COUNT = 7;

constexpr int32_t CONDITION_M_ST = 0;
constexpr int32_t CONDITION_M_END = 1;
constexpr int32_t CONDITION_K_ST = 2;
constexpr int32_t CONDITION_K_END = 3;
constexpr int32_t CONDITION_N_ST = 4;
constexpr int32_t CONDITION_N_END = 5;

std::map<int, std::vector<std::vector<int>>> g_reducescatterFourRankM0Map = {
        {128,
         {{-1, 2560, -1, 7680, -1, 1536}, {-1, 1536, 7680, 2147483647, -1, 1536},
          {1536, 2560, 8704, 2147483647, -1, 1536}, {3584, 2147483647, -1, 4608, -1, 1536},
          {8704, 2147483647, 4608, 5632, -1, 1536}, {2560, 3584, 5632, 2147483647, -1, 1536},
          {-1, 2147483647, -1, 2147483647, 1536, 2147483647}}},
        {256,
         {{1536, 2560, 7680, 8704, -1, 1536}, {2560, 3584, -1, 4608, -1, 1536},
          {2560, 8704, 4608, 5632, -1, 1536}, {3584, 2147483647, 5632, 2147483647, -1, 1536}}}
    };

std::map<int, std::vector<std::vector<int>>> g_reducescatterFourRankUbmovenumMap = {
        {8.0,
         {{-1, 1536, -1, 7168, -1, 1536}, {-1, 1536, -1, 2560, 1536, 3584},
          {1536, 2147483647, -1, 1536, -1, 1536}, {1536, 2560, 1536, 4608, -1, 1536}}},
        {6.0,
         {{-1, 1536, 7168, 2147483647, -1, 1536}, {-1, 1536, 2560, 2147483647, 1536, 3584},
          {-1, 1536, -1, 4608, 3584, 13312}, {1536, 2147483647, -1, 1536, 1536, 13312},
          {1536, 2560, 1536, 4608, 1536, 13312}, {2560, 2147483647, 1536, 4608, -1, 13312},
          {1536, 2560, 4608, 5632, -1, 6144}, {-1, 2147483647, -1, 4608, 13312, 2147483647},
          {5632, 6656, 9728, 2147483647, 13312, 2147483647}}},
        {4.0,
         {{-1, 1536, 4608, 2147483647, 3584, 13312}, {1536, 2560, 4608, 5632, 6144, 13312},
          {2560, 2147483647, 4608, 5632, -1, 13312}, {1536, 2147483647, 5632, 2147483647, -1, 13312},
          {-1, 5632, 4608, 2147483647, 13312, 2147483647}, {5632, 2147483647, 4608, 9728, 13312, 2147483647},
          {6656, 2147483647, 9728, 2147483647, 13312, 2147483647}}}
    };

std::map<int, std::vector<std::vector<int>>> g_reducescatterFourRankPvalueMap = {
        {12,
         {{-1, 1536, -1, 4096, -1, 1536}, {5632, 2147483647, -1, 2560, 3584, 5632}}},
        {1,
         {{-1, 3584, 4096, 2147483647, -1, 1536}, {-1, 3584, 6656, 2147483647, 1536, 3584},
          {4608, 7680, 7680, 2147483647, -1, 3584}, {9728, 2147483647, 8192, 2147483647, -1, 1536},
          {-1, 1536, 6656, 9728, 3584, 2147483647}, {-1, 1536, 9728, 2147483647, 9728, 2147483647},
          {1536, 2560, 7680, 2147483647, 3584, 11264}}},
        {2,
         {{1536, 3584, -1, 4096, -1, 1536}, {-1, 3584, -1, 6656, 1536, 3584},
          {3584, 4608, -1, 2147483647, -1, 2560}, {4608, 7680, 4608, 7680, -1, 3584},
          {7680, 9728, -1, 2147483647, -1, 1536}, {9728, 2147483647, -1, 8192, -1, 1536},
          {-1, 1536, 4608, 6656, 3584, 2147483647}, {-1, 1536, 9728, 2147483647, 3584, 9728},
          {1536, 2560, 5632, 7680, 3584, 2147483647}, {1536, 2560, 7680, 2147483647, 11264, 2147483647}}},
        {4,
         {{3584, 4608, -1, 6144, 2560, 3584}, {4608, 7680, 1536, 4608, -1, 3584},
          {-1, 1536, 1536, 4608, 3584, 2147483647}, {1536, 2560, -1, 4608, 4608, 7680},
          {5632, 6656, 4608, 5632, 3584, 2147483647}, {6656, 8704, 4608, 2147483647, 6656, 2147483647}}},
        {3,
         {{3584, 4608, 6144, 2147483647, 2560, 3584}, {7680, 8704, 4608, 2147483647, 1536, 3584},
          {8704, 2147483647, 5632, 2147483647, 1536, 3584}, {1536, 2560, -1, 4608, 3584, 4608},
          {1536, 2560, 4608, 5632, 3584, 2147483647}, {2560, 5632, 4608, 2147483647, 3584, 2147483647},
          {5632, 6656, 5632, 2147483647, 3584, 2147483647}, {6656, 8704, 4608, 2147483647, 3584, 6656},
          {8704, 2147483647, 4608, 2147483647, 3584, 2147483647}}},
        {8,
         {{4608, 7680, -1, 1536, -1, 3584}, {2560, 5632, -1, 2560, 3584, 7680},
          {4608, 5632, 2560, 4608, 3584, 2147483647}, {5632, 2147483647, 2560, 4608, 3584, 9728}}},
        {6,
         {{7680, 8704, -1, 4608, 1536, 3584}, {8704, 2147483647, -1, 5632, 1536, 3584},
          {-1, 1536, -1, 1536, 3584, 2147483647}, {1536, 2560, 1536, 4608, 7680, 2147483647},
          {2560, 4608, 2560, 4608, 3584, 2147483647}}},
        {10,
         {{1536, 2560, -1, 1536, 7680, 2147483647}}},
        {14,
         {{2560, 5632, -1, 2560, 7680, 2147483647}, {5632, 2147483647, -1, 2560, 5632, 2147483647},
          {5632, 2147483647, 2560, 4608, 9728, 2147483647}}}
    };

std::map<int, std::vector<std::vector<int>>> g_reducescatterPvalueMap = {
    {2,
     {{-1, 1536, -1, 2147483647, -1, 1536}, {1536, 5632, 1536, 2147483647, -1, 1536},
      {-1, 1536, -1, 2147483647, 1536, 2560}, {1536, 5632, 1536, 2147483647, 1536, 2560},
      {5632, 6656, 1536, 2560, -1, 1536}, {5632, 2147483647, 2560, 2147483647, -1, 2560},
      {-1, 4608, 1536, 2560, 2560, 4608}, {-1, 2147483647, 2560, 2147483647, 2560, 2147483647}}},
    {4,
     {{1536, 6656, -1, 1536, -1, 2560}, {5632, 6656, 1536, 2560, 1536, 2560},
      {6656, 2147483647, 1536, 2560, -1, 2560}, {-1, 4608, -1, 1536, 2560, 5632},
      {-1, 4608, 1536, 2560, 4608, 5632}, {4608, 8704, -1, 2560, 2560, 3584},
      {8704, 2147483647, 1536, 2560, 2560, 5632}, {-1, 2560, -1, 2560, 5632, 2147483647},
      {2560, 2147483647, 1536, 2560, 5632, 2147483647}}},
    {6,
     {{6656, 8704, -1, 1536, -1, 2560}}},
    {8,
     {{8704, 2147483647, -1, 1536, -1, 2560}, {4608, 8704, -1, 2560, 3584, 5632},
      {2560, 6656, -1, 1536, 5632, 2147483647}}},
    {10,
     {{8704, 2147483647, -1, 1536, 2560, 5632}}},
    {12,
     {{6656, 2147483647, -1, 1536, 5632, 2147483647}}}
};

std::map<int, std::vector<std::vector<int>>> g_reducescatterCommdatasplitMap = {
    {16,
     {{-1, 9728, -1, 2147483647, -1, 1536}, {9728, 2147483647, -1, 9728, -1, 1536},
      {-1, 2147483647, -1, 2147483647, 1536, 2147483647}}},
    {8,
     {{9728, 2147483647, 9728, 2147483647, -1, 1536}}}
};

std::map<int, std::vector<std::vector<int>>> g_reducescatterUbmovenumMap = {
    {8.0,
     {{-1, 1536, -1, 4096, -1, 1536}, {-1, 1536, 7168, 8704, -1, 1536},
      {1536, 2560, -1, 7680, -1, 1536}, {-1, 2560, 8704, 2147483647, -1, 1536},
      {2560, 2147483647, -1, 1536, -1, 1536}, {3584, 2147483647, 7680, 8704, -1, 1536},
      {6144, 2147483647, 8704, 9728, -1, 1536}, {2560, 3584, 9728, 2147483647, -1, 1536},
      {-1, 1536, -1, 3584, 1536, 2560}, {-1, 1536, -1, 5120, 5632, 7680},
      {1536, 2560, -1, 1536, 1536, 2147483647}, {1536, 2560, 9728, 2147483647, 11264, 2147483647}}},
    {10.0,
     {{-1, 1536, 4096, 7168, -1, 1536}, {1536, 2560, 7680, 8704, -1, 1536},
      {2560, 2147483647, 1536, 7680, -1, 1536}, {2560, 3584, 7680, 8704, -1, 1536},
      {2560, 6144, 8704, 9728, -1, 1536}, {3584, 9728, 9728, 2147483647, -1, 1536},
      {-1, 1536, -1, 3584, 2560, 5632}, {-1, 1536, 3584, 2147483647, 1536, 5632},
      {-1, 1536, -1, 5120, 7680, 13312}, {-1, 1536, 5120, 2147483647, 5632, 13312},
      {-1, 1536, -1, 5120, 13312, 2147483647}, {-1, 1536, 7680, 2147483647, 13312, 2147483647},
      {2560, 2147483647, -1, 1536, 1536, 2147483647}, {1536, 2147483647, 1536, 9728, 1536, 2147483647},
      {1536, 2147483647, 9728, 2147483647, 1536, 11264}, {2560, 2147483647, 9728, 2147483647, 11264, 2147483647}}},
    {20.0,
     {{9728, 2147483647, 9728, 2147483647, -1, 1536}}},
    {6.0,
     {{-1, 1536, 5120, 7680, 13312, 2147483647}}}
};

std::map<int, std::vector<std::vector<int>>> g_reducescatterM0Map = {
    {128,
     {{-1, 5632, -1, 2147483647, -1, 1536}, {5632, 8704, -1, 3584, -1, 1536},
      {-1, 8704, -1, 2147483647, 1536, 7680}, {8704, 2147483647, -1, 2147483647, -1, 7680},
      {-1, 2147483647, -1, 3584, 7680, 2147483647}, {-1, 1536, 3584, 2147483647, 7680, 2147483647},
      {2560, 2147483647, 3584, 2147483647, 7680, 2147483647}}},
    {256,
     {{5632, 8704, 3584, 2147483647, -1, 1536}, {1536, 2560, 3584, 2147483647, 7680, 2147483647}}}
};
}
#endif
