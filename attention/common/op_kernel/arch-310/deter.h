/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deter.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_DETER_H_
#define FLASH_ATTENTION_SCORE_GRAD_DETER_H_

#include "kernel_operator.h"
#include "common.h"

#include <math.h>

namespace commondef {
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_EIGHT = 8;

constexpr int64_t DETER_PREFIX_NUM = 132;
constexpr int64_t DETER_PREFIX_THRESHOLD = 128;

__aicore__ inline int64_t Square(int64_t num) { return num * num; }

__aicore__ inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

__aicore__ inline int64_t AbsCeil(int64_t num1, int64_t num2)
{ 
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (std::abs(num1) + std::abs(num2) - 1) / std::abs(num2);
    return isNegative ? -result : result;
}

struct CoordinateInfo {
    int64_t batchId = 0;
    int64_t n1Idx = 0;
    int64_t s1Idx = 0;
    int64_t s2Idx = 0;

    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t sparseMode = NO_MASK;

    int64_t mOffset = 0;
    int64_t nOffset = 0;

    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

__aicore__ inline void InitCoordinateInfo(int64_t s1Outer, int64_t s2Outer, int64_t mOffset, int64_t nOffset,
                                          CoordinateInfo &coordinateInfo)
{
    coordinateInfo.s1Outer = s1Outer;
    coordinateInfo.s2Outer = s2Outer;
    coordinateInfo.mOffset = mOffset;
    coordinateInfo.nOffset = nOffset;
}

__aicore__ inline void CalDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, CoordinateInfo &coordinate)
{
    k = Min(k, b * m);
    if (j > k) {
        coordinate.batchId = -1;
        return;
    }

    int64_t p = (Ceil<int64_t>(r, m) - 1) * k + j;
    // Determine w and y
    int64_t w = p % b;
    w = (w != 0) ? w : b;
    int64_t y = Ceil<int64_t>(p, b);

    int64_t y1 = y % m;
    y1 = (y1 != 0) ? y1 : m;
    int64_t r1 = r % m;
    r1 = (r1 != 0) ? r1 : m;

    // Calculate x
    int64_t x = y1 + r1 - 1;
    if (x > m) {
        x -= m;
    }

    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

__aicore__ inline void CalCausalG2kSingleBatchDeterIndex(int64_t k, int64_t j, int64_t a, int64_t L1, int64_t offset,
                                                         CoordinateInfo &coordinate)
{
    int64_t x, y = 0;
    if (j % NUM_TWO == 1) { // 奇核分支
        if (a <= L1 - j + 1) {
            y = j + offset;
            x = y + a - 1;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + NUM_TWO * L1 - NUM_TWO * k + 1 - a;
        }
    } else { // 偶核分支
        if (a >= L1 - NUM_TWO * k + 1 + j) {
            y = j + offset;
            x = y + NUM_TWO * L1 - NUM_TWO * k + 1 - a;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + a - 1;
        }
    }
    coordinate.batchId = 0;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalG3kSingleBatchDeterIndex(int64_t j, int64_t a, int64_t L1, int64_t L2, int64_t k, int64_t offset,
                                                         CoordinateInfo &coordinate) {
    int64_t h, I1_min, I1_max, I2_min, I2_max, I3_min, I3_max, bj, cj;
    int64_t x, y;
    if (k % NUM_TWO == 1) {
        h = NUM_THREE * (k + 1) / NUM_TWO;
        I1_min = 1;
        I1_max = h + L2 - 1;
        I2_min = h + L2;
        I2_max = h + L2 + (L1 - k) - 1;
        I3_min = h + L2 + L1 - k;
        I3_max = h + L1 + NUM_TWO * L2 - NUM_TWO;

        bj = (j + (k + 1) / NUM_TWO) % k;
        if (bj == 0) {
            bj = k;
        }   
        cj = h - j - bj;

        if (I1_min <= a && a <= I1_max) {
            if (a <= k - j + 1) {
                y = j + offset;
                x = y + a - 1;
            } else if (a >= NUM_TWO * k - j - bj + NUM_THREE) {
                y = NUM_TWO * k + cj + offset;
                x = y + (I1_max - a);
            } else {
                y = k + bj + offset;
                x = y + (a - (k - j + NUM_TWO));
            }
        } else if (I2_min <= a && a <= I2_max) {
            int64_t mod_val = (a - h - L2 + j) % (L1 - k);
            int64_t tmp = mod_val + (k - j + 1);
            if (mod_val == 0) {
                tmp = (L1 - k) + (k - j + 1);
            }
            y = j + offset;
            x = y + tmp - 1;
        } else { // a in I3
            int64_t mod_val = (a - h - L2 - L1 + k + bj) % (L2 + k - 1);
            int64_t tmp = mod_val + (k - bj + 1);
            if (mod_val == 0) {
                tmp = (L2 + k - 1) + (k - bj + 1);
            }
            y = k + bj + offset;
            x = y + tmp - 1;
        }
    } else { // Even k branch
        h = NUM_THREE * k / NUM_TWO + NUM_TWO;
        I1_min = 1;
        I1_max = h + L2 - 1;
        I2_min = h + L2;
        I2_max = h + L2 + (L1 - k) - 1;
        I3_min = h + L2 + L1 - k;
        I3_max = h + L1 + NUM_TWO * L2 - NUM_TWO;

        bj = (j + k / NUM_TWO) % k;
        if (bj == 0) {
            bj = k;
        }
        if (j <= k / NUM_TWO) {
            cj = h - j - bj;
        } else {
            cj = (h - 1) - j - bj;
        }

        if (I1_min <= a && a <= I1_max) {
            if (a <= k - j + 1) {
                y = j + offset;
                x = y + a - 1;
            } else if (j > k / NUM_TWO && (k - j + NUM_TWO) <= a && a <= (NUM_TWO * k - j - bj + NUM_TWO)) {
                y = k + bj + offset;
                x = y + (a - (k - j + NUM_TWO));
            } else if (j <= k / NUM_TWO && a == (k - j + NUM_TWO)) {
                coordinate.batchId = -1;
                return;
            } else if (j <= k / NUM_TWO && (k - j + NUM_THREE) <= a && a <= (NUM_TWO * k - j - bj + NUM_THREE)) {
                y = k + bj + offset;
                x = y + (a - (k - j + NUM_THREE));
            } else {
                y = NUM_TWO * k + cj + offset;
                x = y + (I1_max - a);
            }
        } else if (I2_min <= a && a <= I2_max) {
            int64_t mod_val = (a - h - L2 + j) % (L1 - k);
            int64_t tmp = mod_val + (k - j + 1);
            if (mod_val == 0) {
                tmp = (L1 - k) + (k - j + 1);
            }
            y = j + offset;
            x = y + tmp - 1;
        } else {
            int64_t mod_val = (a - h - L2 - L1 + k + bj) % (L2 + k - 1);
            int64_t tmp = mod_val + (k - bj + 1);
            if (mod_val == 0) {
                tmp = (L2 + k - 1) + (k - bj + 1);
            }
            y = k + bj + offset;
            x = y + tmp - 1;
        }
    }
    coordinate.batchId = 0;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalNoRecSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                           CoordinateInfo &coordinate)
{
    coordinate.batchId = -1;

    // 核心编号超过 ⌊n/2⌋ + 1 时无任务可处理
    if (j > (n / NUM_TWO) + 1) {
        return;
    }

    int64_t x, y = 0;
    if (j % NUM_TWO == 1) { // 奇核
        if (r + j <= n + 1) {
            x = r + j - 1;
            y = j;
        } else {
            x = NUM_TWO * n + NUM_TWO - j - r;
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    } else { // 偶核
        if (j <= r + 1 - (n % NUM_TWO)) {
            x = n + j - r - 1 + n % NUM_TWO;
            y = j;
        } else {
            x = n + NUM_TWO + r - j - (n % NUM_TWO);
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    }

    // 检查 (x, y) 是否在下三角范围内
    if (y >= 1 && y <= m && y <= x && x <= m) {
        coordinate.batchId = 0;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    }
    return;
}

// 矩形
__aicore__ inline void CalCausalRecSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                         CoordinateInfo &coordinate)
{
    int64_t x, y = 0;
    // Choose method based on core count
    if (NUM_TWO * k < m + 1 && k < n) {
        CalCausalG2kSingleBatchDeterIndex(k, j, r, m, 0, coordinate);
        x = coordinate.s1Idx;
        y = coordinate.s2Idx;
    } else {
        CalCausalNoRecSingleBatchDeterIndex(k, m, m, j, r, coordinate);
        if (coordinate.batchId != -1) {
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
        } else {
            x = m + 1;
            y = m + 1;
        }
    }

    // Check if the computed position is within the valid submatrix
    if (y >= 1 && y <= n && y <= x && x <= m) {
        coordinate.batchId = 0;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

// 计算单batch切分
__aicore__ inline void CalCausalSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r, CoordinateInfo &coordinate)
{
    // 0. special case
    if (k >= (n / NUM_TWO) + 1) {
        CalCausalRecSingleBatchDeterIndex(k, m, n, j, r, coordinate);
        return;
    }

    // 1. compute ell and t1, t2, t3
    int64_t ell = n % k;
    int64_t t1 = n / (NUM_TWO * k);
    int64_t t2 = 0;
    int64_t t3 = (n / k) % NUM_TWO;

    int64_t rm, L1, a, offset;
    // 2. 2k-block groups
    int64_t bound1 = (NUM_TWO * m + 1) * t1 - NUM_TWO * k * t1 * t1;
    if (r <= bound1) {
        // solve for group index i
        int64_t disc = Square((NUM_TWO * m + 1)) - NUM_EIGHT * k * r;
        int64_t i = Ceil<int64_t>(((NUM_TWO * m + 1) - sqrt(disc)), (NUM_FOUR * k));
        rm = (NUM_TWO * m + 1) * (i - 1) - NUM_TWO * k * (i - 1) * (i - 1);
        offset = NUM_TWO * k * (i - 1);
        L1 = m - NUM_TWO * k * (i - 1);
        a = r - rm;
        CalCausalG2kSingleBatchDeterIndex(k, j, a, L1, offset, coordinate);
        return;
    } else {
        rm = bound1;
        offset = NUM_TWO * k * t1;
    }

    // 4. remaining block
    int64_t rem = t3 * k + ell;
    coordinate.batchId = -1;
    if (rem > 0) {
        L1 = m - n + rem;
        a = r - rm;
        if (t3 == 0 && j <= rem) {
            coordinate.batchId = 0;
            coordinate.s2Idx = offset + j;
            coordinate.s1Idx = coordinate.s2Idx + a - 1;
            return;
        } else if (t3 == 1) {
            CalCausalRecSingleBatchDeterIndex(k, L1, L1, j, a, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            
            int64_t shift = NUM_TWO * k * t1 + NUM_THREE * k * t2;
            coordinate.s1Idx += shift;
            coordinate.s2Idx += shift;
            return;
        } else {
            return;
        }   
    }
    return;
}

__aicore__ inline void CalCausalIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                      CoordinateInfo &coordinate)
{
    // 1. b1, b2, rm1
    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t delta = m - n;
    int64_t sizeTri = n * (m + delta + 1) / NUM_TWO;
    int64_t rm1 = b1 * sizeTri;

    int64_t w, x, y = 0;
    // --- 第一段：完整矩形 ---
    if (r <= rm1) {
        int64_t a = r % sizeTri;
        a = a != 0 ? a : sizeTri;
        w = Ceil<int64_t>(r, sizeTri) + b1 * (j - 1);
        int64_t n1 = n / NUM_TWO * NUM_TWO;
        int64_t L = NUM_TWO * m - n1 + 1;
        int64_t rmLocal = (n1 * L) / NUM_TWO;
        if (a <= rmLocal) {
            y = Ceil<int64_t>(a, L);
            int64_t r1 = a % L != 0 ? a % L : L;
            x = r1 + y - 1;
            if (x > m) {
                x = NUM_TWO * m + 1 - x;
                y = n1 + 1 - y;
            }
        } else {
            int64_t a1 = a - rmLocal;
            y = n;
            x = a1 - 1 + y;
        }
        w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
        w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    // 2. 行块压平分给 floor(b2/2) 个 batch
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t a2 = r - rm1;
    int64_t half_ = b2 / NUM_TWO;
    int64_t rm2 = (NUM_TWO * m - t * k + 1) * t * half_;
    if (a2 >= 1 && a2 <= rm2) {
        // 调用 dense 子块
        CalDenseIndex(k, NUM_TWO * m - t * k + 1, t * k, half_, j, a2, coordinate);
        int64_t wSub = coordinate.batchId;
        int64_t xSub = coordinate.s1Idx;
        int64_t ySub = coordinate.s2Idx;
        // 新增范围检查
        int64_t max_x = NUM_TWO * m - t * k + 1;
        int64_t max_y = t * k;
        int64_t max_w = half_;
        if (xSub >= 1 && xSub <= max_x && ySub >= 1 && ySub <= max_y && wSub >= 1 && wSub <= max_w) {
            // 计算映射
            if (xSub - ySub <= m - t * k) {
                w = NUM_TWO * wSub - 1 + b1 * k;
                x = m + 1 - xSub;
                y = t * k + 1 - ySub;
            } else {
                w = NUM_TWO * wSub + b1 * k;
                x = xSub - m + t * k - 1;
                y = ySub;
            }
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
        } else {
            coordinate.batchId = -1;
        }
        return;
    }

    // 3. 剩余，分两种：b2 mod2 == 1 或 else
    int64_t a3 = r - rm1 - rm2;
    int64_t rm3 = 0;
    if (b2 % NUM_TWO == 1) {
        int64_t t1 = n / (NUM_TWO * k);
        int64_t t3 = t % NUM_TWO;
        // 奇数 b2
        if (t3 == 1) {
            int64_t m1 = m - t1 * NUM_TWO * k;
            rm3 = (m + m1 + 1) * t1;
            rm3 += ell == 0 ? m1 : Max(m1, NUM_TWO * m1 - NUM_TWO * k + 1);
            if (a3 >= 1 && a3 <= rm3) {
                CalCausalSingleBatchDeterIndex(k, m, n, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = b;
                    return;
                } else {
                    return;
                }
            }
            b -= 1;
            b2 -= 1;
        } else {
            // 偶数 b2
            rm3 = (NUM_TWO * m - t * k + 1) * t / NUM_TWO;
            if (a3 >= 1 && a3 <= rm3) {
                CalCausalSingleBatchDeterIndex(k, m, t * k, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = b;
                    return;
                } else {
                    return;
                }
            }
        }
    }

    // 4. 零碎余数 ℓ 部分
    int64_t a4 = a3 - rm3;
    int64_t p = Ceil<int64_t>(ell, NUM_TWO);
    int64_t ell1 = ell + 1 - (ell % NUM_TWO);
    int64_t block = (b2 * p) / k;
    int64_t res0 = (b2 * p) % k;
    // 情况 A
    if (a4 > block * (ell1 + NUM_TWO * delta) && res0 <= k / NUM_TWO) {
        int64_t offs = a4 - block * (ell1 + NUM_TWO * delta);
        // 前 res0 cores
        if (j >= 1 && j <= res0) {
            int64_t limit = (res0 - j) / b2 + Ceil<int64_t>(ell1, NUM_TWO) + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
                w = w != 0 ? w : b2;
                y = p - (res0 - j) / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    coordinate.batchId = b1 * k + w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }

        // 后 res0 cores
        if (k - res0 + 1 <= j && j <= k) {
            int64_t idx = j - (k - res0 + 1);
            int64_t limit = (ell1 - 1) / NUM_TWO - idx / b2 + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + k + 1 - j) % b2;
                w = w != 0 ? w : b2;
                y = p + 1 + idx / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    coordinate.batchId = b1 * k + w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }
        coordinate.batchId = -1;
        return;
    } else {
        // 情况 B
        w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
        w = w != 0 ? w : b2;
        int64_t g = Ceil<int64_t>((k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j), b2);
        if (g >= 1 && g <= p) {
            int64_t a5 = a4 % (ell1 + NUM_TWO * delta);
            a5 = a5 != 0 ? a5 : (ell1 + NUM_TWO * delta);
            int64_t x0, y0 = 0;
            if (g % NUM_TWO == 1) {
                if (a5 <= ell - g + 1 + delta) {
                    x0 = g + a5 - 1;
                    y0 = g;
                } else {
                    x0 = NUM_TWO * ell + NUM_TWO * delta + NUM_TWO - g - a5;
                    y0 = ell + 1 + (ell % NUM_TWO) - g;
                }
            } else {
                if (a5 >= (g + 1 + delta - (ell % NUM_TWO))) {
                    x0 = g + ell + NUM_TWO * delta + 1 - (ell % NUM_TWO) - a5;
                    y0 = g;
                } else {
                    x0 = a5 + ell - g + (ell % NUM_TWO);
                    y0 = ell + 1 + (ell % NUM_TWO) - g;
                }
            }

            if (y0 >= 1 && y0 <= ell && y0 <= x0 && x0 <= ell + delta) {
                coordinate.batchId = b1 * k + w;
                coordinate.s1Idx = x0 + t * k;
                coordinate.s2Idx = y0 + t * k;
                return;
            }
        }
    }
    coordinate.batchId = -1;
    return;
}

struct BandInfo {
    int64_t k = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
    int64_t b = 0;
    int64_t b1 = 0;
    int64_t b2 = 0;
    int64_t L1 = 0;
    int64_t L2 = 0;
    int64_t L3 = 0;
    int64_t nSeg = 0;
    int64_t R1 = 0;
    int64_t R2 = 0;
    int64_t R3 = 0;
    int64_t Rm = 0;
    int64_t rm = 0;
    int64_t a = 0;
};

__aicore__ inline void GenBandInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, BandInfo &bandInfo) {
    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;

    // 1. initial b1, b2
    bandInfo.b1 = b / k;
    bandInfo.b2 = b % k;
    // p+q>m的分支
    if (p + q > m) {
        // 2. lengths
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(m - 1, n - q);
        // 3. redefine nSeg
        bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        // 4. segment sizes
        bandInfo.R1 = (p + m - 1) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = m * bandInfo.L2;
        bandInfo.R3 = (NUM_TWO * m - 1 - bandInfo.L3) * bandInfo.L3 / NUM_TWO;
        bandInfo.Rm = bandInfo.R1 + bandInfo.R2 + bandInfo.R3;
        // 5. rm = b1 * Rm
        bandInfo.rm = bandInfo.b1 * bandInfo.Rm;
    } else {
        // p+q<=m的分支
        // 2. lengths
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        // 3. redefine nSeg
        bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        // 4. segment sizes
        bandInfo.R1 = (NUM_TWO * p - NUM_TWO + q) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = (p + q - 1) * bandInfo.L2;
        bandInfo.R3 = (p + q - NUM_TWO) * bandInfo.L3 - (bandInfo.L3 * (bandInfo.L3 - 1)) / NUM_TWO;
        bandInfo.Rm = bandInfo.R1 + bandInfo.R2 + bandInfo.R3;
        // 5. rm = b1 * Rm
        bandInfo.rm = bandInfo.b1 * bandInfo.Rm;
    }
}

__aicore__ inline void CalBandIndex(const BandInfo &bandInfo, int64_t j, int64_t r, CoordinateInfo &coordinate)
{
    int64_t w, x, y = 0;
    coordinate.batchId = -1;

    int64_t k = bandInfo.k;
    int64_t m = bandInfo.m;
    int64_t n = bandInfo.n;
    int64_t p = bandInfo.p;
    int64_t q = bandInfo.q;
    int64_t b = bandInfo.b;
    int64_t b1 = bandInfo.b1;
    int64_t b2 = bandInfo.b2;
    int64_t L1 = bandInfo.L1;
    int64_t L2 = bandInfo.L2;
    int64_t L3 = bandInfo.L3;
    int64_t nSeg = bandInfo.nSeg;
    int64_t R1 = bandInfo.R1;
    int64_t R2 = bandInfo.R2;
    int64_t R3 = bandInfo.R3;
    int64_t Rm = bandInfo.Rm;
    int64_t rm = bandInfo.rm;
    int64_t a;
    // p+q>m的分支
    if (p + q > m) {
        if (r <= rm) {
            a = r % Rm;
            w = Ceil<int64_t>(r, Rm) + b1 * (j - 1);
            if (a == 0) {
                a = Rm;
            }

            // subsegment selection
            if (a <= R1) {
                int64_t L11 = L1 / NUM_TWO * NUM_TWO;
                int64_t L = NUM_TWO * p + L11 - 1;
                int64_t local_round = L11 * L / NUM_TWO;
                if (a <= local_round) {
                    y = Ceil<int64_t>(a, L);
                    int64_t r1 = a % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = p + y - r1;
                    if (x < 1) {
                        y = L11 + 1 - y;
                        x = 1 - x;
                    }
                } else {
                    x = a - local_round;
                    y = L1;
                }
            } else if (a <= R1 + R2) {
                int64_t a2 = a - R1;
                y = Ceil<int64_t>(a2, m);
                x = a2 % m;
                if (x == 0) {
                    x = m;
                }
                y = y + L1;
            } else {
                int64_t a3 = a - R1 - R2;

                int64_t L31 = L3 / NUM_TWO * NUM_TWO;
                int64_t L = NUM_TWO * m - L31 - 1;
                int64_t local_round = L31 * L / NUM_TWO;
                if (a3 <= local_round) {
                    y = Ceil<int64_t>(a3, L);
                    int64_t r1 = a3 % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = y + r1;
                    if (x > m) {
                        y = L31 + 1 - y;
                        x = NUM_TWO * m + 1 - x;
                    }
                } else {
                    x = m - a3 + local_round + 1;
                    y = L3;
                }
                y = y + L1 + L2;
            }
            w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
            w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }

        // --- second phase: overflow rows ---
        a = r - rm;

        CalDenseIndex(k, m, n, b2, j, a, coordinate);
        w = coordinate.batchId;
        if (w != -1) {
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
            if (y <= m - p && x >= p + y) {
                return;
            }
            if (y > L1 + L2 && x <= y - L1 - L2) {
                return;
            }
            if (w > 0 && w < b2 + 1 && x > 0 && x < m + 1 && y > 0 && y < n + 1) {
                coordinate.batchId = b1 * k + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            }
        }
        return;
    }

    // p+q<=m的分支
    if (L3 == 0) {
        m = p + q + L2 - NUM_TWO;
    }
    if (r <= rm) {
        a = r % Rm;
        w = Ceil<int64_t>(r, Rm) + b1 * (j - 1);
        if (a == 0) {
            a = Rm;
        }
        
        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, (p + q - 1));
            x = a2 % (p + q - 1) + (y - 1);
            if (x == y - 1) {
                x = (p + q - 1) + y - 1;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * (p + q) - L31 - NUM_THREE;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1 + 1 + m - (p + q);
                if (x > m) {
                    y = L31 + 1 - y;
                    x = NUM_TWO * m + 1 - x;
                }
            } else {
                x = m - a3 + local_round + 1;
                y = L3;
            }
            y = y + L1 + L2;
        }

        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    if (b2 == 0) {
        coordinate.batchId = -1;
        return;
    }

    // --- second phase: overflow rows ---
    a = r - rm;
    int64_t seg = p + q - 1;
    int64_t a1 = Ceil<int64_t>(a, seg);
    int64_t a2 = a % seg;
    a2 = a2 == 0 ? seg : a2;

    // if no L3 or nSeg - m < 1
    if (L3 == 0 || nSeg - m < 1) {
        int64_t idx = (a1 - 1) * k + j;
        w = Ceil(idx, nSeg);
        y = idx % nSeg;
        if (y == 0) {
            y = nSeg;
        }

        x = y + a2 - q;
        if (x >= 1 && x <= m) {
            coordinate.batchId = w + b1 * k;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }
        return;
    }

    // else L3>0 and nSeg-m >=1
    if (L3 > 0 && nSeg - m >= 1) {
        y = (a1 - 1) * k + j;
        x = y + a2 - q;
        if (x < 1) {
            coordinate.batchId = b;
            coordinate.s1Idx = x + m;
            coordinate.s2Idx = y + m;
            return;
        }

        w = Ceil<int64_t>(x, m);
        x = x % m;
        if (x == 0) {
            x = m;
        }
        y = x + q - a2;
        if (w == b2 && y > m) {
            return;
        }
        if (y >= 1 && y <= n) {
            coordinate.batchId = w + b1 * k;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }
        return;
    }
    return;
}

template <bool isAlign = false>
__aicore__ inline int64_t GetPrefixByBidx(const __gm__ uint8_t *actualSeqQlenAddr,
                                          const __gm__ uint8_t *actualSeqKvlenAddr,
                                          const int64_t (&prefix)[DETER_PREFIX_NUM], int64_t bIdx, int64_t step)
{
    int64_t w = bIdx / step;

    int64_t prefixNum = prefix[w];
    if (bIdx % step > 0) {
        w = w * step;
        int64_t lastSeqQLen = w == 0 ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[w - 1];
        int64_t lastSeqKvLen = w == 0 ? 0 : ((__gm__ int64_t *)actualSeqKvlenAddr)[w - 1];
        int64_t currentSeqQLen, currentSeqKvLen;
        while (w < bIdx) {
            currentSeqQLen = ((__gm__ int64_t *)actualSeqQlenAddr)[w];
            currentSeqKvLen = ((__gm__ int64_t *)actualSeqKvlenAddr)[w];
            prefixNum += (currentSeqQLen - lastSeqQLen) *
                         (isAlign ? AlignTo16(currentSeqKvLen - lastSeqKvLen) : (currentSeqKvLen - lastSeqKvLen));
            w += 1;
            lastSeqQLen = currentSeqQLen;
            lastSeqKvLen = currentSeqKvLen;
        }
    }
    return prefixNum;
}

__aicore__ inline void GetSeqQlenKvlenByBidx(const __gm__ uint8_t *actualSeqQlenAddr,
                                             const __gm__ uint8_t *actualSeqKvlenAddr, int64_t bIdx,
                                             int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx - 1];
    }
}

__aicore__ inline void CalVirtualIndex(int64_t flag, int64_t &m, int64_t &n)
{
    if (flag == 0) {
        m = NUM_TWO * m - n + 1;
    } else if (flag == 1) {
        m = m - (n + 1) / NUM_TWO + 1;
        n = n / NUM_TWO;
    } else {
        m = m - n / NUM_TWO;
        n = (n + 1) / NUM_TWO;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool isDense>
__aicore__ inline void UpdateMNPQ(int64_t actualCalcS1Token, int64_t actualCalcS2Token, CoordinateInfo &coordinateInfo,
                                  int64_t &actualM, int64_t &actualN)
{
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    int64_t actualS1Len = coordinateInfo.actualS1Len;
    int64_t actualS2Len = coordinateInfo.actualS2Len;
    if (coordinateInfo.sparseMode == BAND) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }

    int64_t m = coordinateInfo.s1Outer;
    int64_t n = coordinateInfo.s2Outer;
    int64_t p = Ceil<int64_t>(actualCalcS1Token, CUBE_BASEM) + 1;
    int64_t q = Ceil<int64_t>(actualCalcS2Token, CUBE_BASEN) + 1;
    p = p > m ? m : p;
    q = q > n ? n : q;

    // 负数场景变换
    if (p < 0) {
        coordinateInfo.mOffset = 0;
        coordinateInfo.nOffset = -p;
        n = n + p;
        q = p + q;
        p = 1;
    } else if (q < 0) {
        coordinateInfo.mOffset = -q;
        coordinateInfo.nOffset = 0;
        m = m + q;
        p = p + q;
        q = 1;
    } else {
        coordinateInfo.mOffset = 0;
        coordinateInfo.nOffset = 0;
    }

    if (p + q <= m) {
        int64_t L1 = q - 1;
        int64_t L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        int64_t L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));

        if (L3 == 0) {
            actualM = p + q + L2 - NUM_TWO;
        } else {
            actualM = m;
        }
        actualN = L1 + L2 + L3;
    } else {
        actualM = m;
        actualN = Min(m - 1 + q, n);
    }

    coordinateInfo.m = actualM;
    coordinateInfo.n = actualN;
    coordinateInfo.p = p;
    coordinateInfo.q = q;

    if constexpr (!isDense) {
        return;
    }

    if (p + q <= actualM) {
        if (actualN >= actualM) {
            actualN = p + q - 1;
        } else {
            actualM = p + q - 1;
        }
    } else {
        if (p + q <= actualN) {
            actualN = p + q - 1;
        }
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const uint8_t DETER_SPARSE_TYPE = DETER_DENSE>
__aicore__ inline void
CalTNDDenseIndex(const __gm__ uint8_t *actualSeqQlenAddr, const __gm__ uint8_t *actualSeqKvlenAddr,
                 const int64_t (&prefix)[DETER_PREFIX_NUM], int64_t deterMaxRound, int64_t b, int64_t N1, int64_t j,
                 int64_t r, uint8_t flag, int64_t step, CoordinateInfo &coordinateInfo)
{
    coordinateInfo.batchId = -1;
    if (r > deterMaxRound) {
        return;
    }

    int64_t ID = (j - 1) * deterMaxRound + r;
    int64_t w = 0;
    while ((w + 1) * step < b && ID > prefix[w + 1] * N1) {
        w += 1;
    }
    int64_t delta = ID - prefix[w] * N1;

    w = w * step;
    if (w >= b) {
        return;
    }

    int64_t m, n, p, q;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, w, actualS1Len, actualS2Len);

    m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
    n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
    coordinateInfo.actualS1Len = actualS1Len;
    coordinateInfo.actualS2Len = actualS2Len;
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
    int64_t actualCalcS1Token = coordinateInfo.p;
    int64_t actualCalcS2Token = coordinateInfo.q;

    if constexpr (DETER_SPARSE_TYPE == DETER_BAND) {
        UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, true>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m, n);
        p = coordinateInfo.p;
        q = coordinateInfo.q;
    }

    if constexpr (DETER_SPARSE_TYPE == DETER_CAUSAL) {
        CalVirtualIndex(flag, m, n);
    }

    if (unlikely(step > 1)) {
        int64_t batchBaseNum = m * n * N1;
        while (delta > batchBaseNum && w < b) {
            delta = delta - batchBaseNum;
            w += 1;
            if (w >= b) {
                return;
            }
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, w, actualS1Len, actualS2Len);
            m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            coordinateInfo.s1Outer = m;
            coordinateInfo.s2Outer = n;
            if constexpr (DETER_SPARSE_TYPE == DETER_CAUSAL) {
                CalVirtualIndex(flag, m, n);
            } else if constexpr (DETER_SPARSE_TYPE == DETER_BAND) {
                UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, true>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m, n);
                p = coordinateInfo.p;
                q = coordinateInfo.q;
            }

            batchBaseNum = m * n * N1;
        }
        coordinateInfo.actualS1Len = actualS1Len;
        coordinateInfo.actualS2Len = actualS2Len;
    }

    int64_t currentBaseNum = m * n;
    int64_t batchId = w + 1;
    int64_t deltaN = (delta - 1) / currentBaseNum + 1;
    delta = delta % currentBaseNum;
    delta = delta != 0 ? delta : currentBaseNum;

    int64_t g = Gcd(m, deterMaxRound);
    int64_t t1 = deterMaxRound / g;
    int64_t t2 = m / g;

    int64_t x = ((delta - 1) % m) + 1;
    int64_t y = (delta - 1) / m + 1;
    if (t1 < n) {
        int64_t n1 = n % t1;
        n1 = n1 == 0 ? t1 : n1;
        if (y <= n - n1) {
            int64_t delta_adj = Ceil<int64_t>(y, t1);
            delta += delta_adj;
            if (delta > delta_adj * t2 * deterMaxRound) {
                delta -= t2 * deterMaxRound;
            }
            x = ((delta - 1) % m) + 1;
            y = (delta - 1) / m + 1;
        }
    }
    coordinateInfo.batchId = (batchId - 1) * N1 + deltaN;
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalPosWholeBatch(int64_t m, int64_t n, int64_t a, CoordinateInfo &coordinateInfo)
{
    int64_t n1 = n / NUM_TWO * NUM_TWO;
    int64_t L = NUM_TWO * m - n1 + 1;
    int64_t rm_local = (n1 * L) / NUM_TWO;
    int64_t x, y;
    if (a <= rm_local) {
        y = Ceil<int64_t>(a, L);
        int64_t r1 = a % L;
        r1 = r1 != 0 ? r1 : L;
        x = r1 + y - 1;
        if (x > m) {
            x = NUM_TWO * m + 1 - x;
            y = n1 + 1 - y;
        }
    } else {
        int64_t a1 = a - rm_local;
        y = n;
        x = a1 - 1 + y;
    }
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
    return;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void
CalTNDCausalIndex(const __gm__ uint8_t *actualSeqQlenAddr, const __gm__ uint8_t *actualSeqKvlenAddr,
                  const int64_t (&prefix0)[DETER_PREFIX_NUM], const int64_t (&prefix1)[DETER_PREFIX_NUM],
                  const int64_t (&prefix2)[DETER_PREFIX_NUM], int64_t b, int64_t N1, int64_t k, int64_t j, int64_t r,
                  int64_t step, CoordinateInfo &coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t N10 = N1 / k;
    int64_t N11 = N1 % k / NUM_TWO;
    int64_t R01 = prefix0[maxRoundIndex];
    int64_t R02 = prefix0[maxRoundIndex + 1];
    int64_t R0 = R01 + R02;
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t R2 = prefix2[maxRoundIndex];
    coordinateInfo.batchId = -1;
    if (r <= R01) {
        int64_t a_judge = Ceil<int64_t>(r * NUM_TWO, N10);
        int64_t w = 0;
        while ((w + 1) * step < b && a_judge > prefix0[w + 1]) {
            w += 1;
        }
        int64_t batch_id = w * step;
        int64_t a = r - prefix0[w] * N10 / NUM_TWO;

        if (batch_id >= b) {
            return;
        }

        int64_t actualS1Len, actualS2Len, m, n;
        int64_t round_batch;

        GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, actualS1Len, actualS2Len);
        m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        round_batch = (NUM_TWO * m - n + 1) * n / NUM_TWO;

        while (a > round_batch * N10) {
            batch_id += 1;
            if (batch_id >= b) {
                return;
            }
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, actualS1Len, actualS2Len);
            m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            a = a - round_batch * N10;
            round_batch = (NUM_TWO * m - n + 1) * n / NUM_TWO;
        }

        int64_t a0 = a % round_batch;
        a0 = a0 != 0 ? a0 : round_batch;
        CalCausalPosWholeBatch(m, n, a0, coordinateInfo);
        int64_t x = coordinateInfo.s1Idx;
        int64_t y = coordinateInfo.s2Idx;

        w = (a - 1) / round_batch * k + j;
        // b1组，每组k个batch行交错
        w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        batch_id = batch_id * N1 + w;
        coordinateInfo.batchId = batch_id;
        coordinateInfo.s1Outer = m;
        coordinateInfo.s2Outer = n;
        coordinateInfo.actualS1Len = actualS1Len;
        coordinateInfo.actualS2Len = actualS2Len;
        return;
    } else if (r > R01 && r <= R0) {
        int64_t a = r - R01;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, R02, b,
                                                               N11, j, a, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m = coordinateInfo.s1Outer;
        int64_t n = coordinateInfo.s2Outer;
        int64_t x = coordinateInfo.s1Idx;
        int64_t y = coordinateInfo.s2Idx;

        int64_t batch_id1 = Ceil<int64_t>(coordinateInfo.batchId, N11);
        int64_t batch_id2 = coordinateInfo.batchId % N11;
        batch_id2 = batch_id2 != 0 ? batch_id2 : N11;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
            batch_id2 = NUM_TWO * batch_id2 - 1;
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
            batch_id2 = NUM_TWO * batch_id2;
        }
        coordinateInfo.batchId = (batch_id1 - 1) * N1 + N10 * k + batch_id2;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (R0 < r && r <= R0 + R1) {
        int64_t a = r - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1, b, 1,
                                                               j, a, 1, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m = coordinateInfo.s1Outer;
        int64_t n = coordinateInfo.s2Outer;
        int64_t x = coordinateInfo.s1Idx;
        int64_t y = coordinateInfo.s2Idx;

        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
        }
        coordinateInfo.batchId = coordinateInfo.batchId * N1;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else {
        int64_t a = r - R1 - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix2, R2, b, 1,
                                                               j, a, NUM_TWO, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        coordinateInfo.s1Idx += coordinateInfo.s2Outer / NUM_TWO;

        coordinateInfo.batchId = coordinateInfo.batchId * N1;
        coordinateInfo.s1Idx = coordinateInfo.s1Idx;
        return;
    }
}

__aicore__ inline int64_t BinarySearch(const int64_t (&prefix)[DETER_PREFIX_NUM], int64_t b, int64_t rThreshold,
                                       int64_t step)
{
    int64_t w = 0;
    while ((w + 1) * step < b && rThreshold > prefix[w + 1]) {
        w += 1;
    }
    return w;
}

__aicore__ inline void CalPosWholeBatch(int64_t m, int64_t n, int64_t p, int64_t q, int64_t a,
                                        CoordinateInfo &coordinateInfo)
{
    int64_t L1, L2, L3, R1, R2, R3, Rm;
    int64_t x, y;
    if (p + q > m) {
        L1 = m - p;
        L2 = p + q - m;
        L3 = Min(m - 1, n - q);
        R1 = (p + m - 1) * L1 / NUM_TWO;
        R2 = m * L2;
        R3 = (NUM_TWO * m - 1 - L3) * L3 / NUM_TWO;
        Rm = R1 + R2 + R3;

        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2 && m != 0) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, m);
            x = a2 % m;
            if (x == 0) {
                x = m;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * m - L31 - 1;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1;
                if (x > m) {
                    y = L31 + 1 - y;
                    x = NUM_TWO * m + 1 - x;
                }
            } else {
                x = m - a3 + local_round + 1;
                y = L3;
            }
            y = y + L1 + L2;
        }
        coordinateInfo.s1Idx = x;
        coordinateInfo.s2Idx = y;
        return;
    } else {
        L1 = q - 1;
        L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        R1 = (NUM_TWO * p - NUM_TWO + q) * L1 / NUM_TWO;
        R2 = (p + q - 1) * L2;
        R3 = (p + q - NUM_TWO) * L3 - (L3 * (L3 - 1)) / NUM_TWO;
        Rm = R1 + R2 + R3;
        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, (p + q - 1));
            x = a2 % (p + q - 1) + (y - 1);
            if (x == y - 1) {
                x = (p + q - 1) + y - 1;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * (p + q) - L31 - NUM_THREE;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1 + 1 + m - (p + q);
                if (x > m) {
                    y = L31 + 1 - y;
                    x = NUM_TWO * m + 1 - x;
                }
            } else {
                x = m - a3 + local_round + 1;
                y = L3;
            }
            y = y + L1 + L2;
        }
        coordinateInfo.s1Idx = x;
        coordinateInfo.s2Idx = y;
        return;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void
CalTNDBandIndex(const __gm__ uint8_t *actualSeqQlenAddr, const __gm__ uint8_t *actualSeqKvlenAddr,
                const int64_t (&prefix0)[DETER_PREFIX_NUM], const int64_t (&prefix1)[DETER_PREFIX_NUM], int64_t b,
                int64_t N1, int64_t k, int64_t j, int64_t r, int64_t step, CoordinateInfo &coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t R0 = prefix0[maxRoundIndex];
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t N11 = N1 / k;
    int64_t N12 = N1 % k;

    int64_t m, n, p, q, x, y;
    if (r <= R0) {
        int64_t batch_id = BinarySearch(prefix0, b, r, step);
        int64_t a = r - prefix0[batch_id];
        int64_t round_batch = 0;
        batch_id = batch_id * step - 1;
        int64_t actualCalcS1Token = coordinateInfo.p;
        int64_t actualCalcS2Token = coordinateInfo.q;
        do {
            batch_id++;
            a = a - round_batch * N11;
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, coordinateInfo.actualS1Len,
                                  coordinateInfo.actualS2Len);
            coordinateInfo.s1Outer = (coordinateInfo.actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            coordinateInfo.s2Outer = (coordinateInfo.actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            // 重新计算m、n、p、q
            UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, false>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m, n);
            p = coordinateInfo.p;
            q = coordinateInfo.q;
            round_batch = m * n - (m - p) * (m - p + 1) / NUM_TWO - (n - q) * (n - q + 1) / NUM_TWO;
        } while (batch_id < b && a > round_batch * N11);

        int64_t a0 = a % round_batch;
        a0 = a0 != 0 ? a0 : round_batch;
        CalPosWholeBatch(m, n, p, q, a0, coordinateInfo);
        y = coordinateInfo.s2Idx;
        // 错位
        int64_t w = (a - 1) / round_batch * k + j;
        // b1组，每组k个batch行交错
        coordinateInfo.batchId = batch_id * N1 + w;
        return;
    } else if (R0 < r && r <= R0 + R1) {
        int64_t a = r - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_BAND>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1, b, N12,
                                                             j, a, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        m = coordinateInfo.m;
        n = coordinateInfo.n;
        p = coordinateInfo.p;
        q = coordinateInfo.q;
        x = coordinateInfo.s1Idx;
        y = coordinateInfo.s2Idx;
        int64_t batch_id = coordinateInfo.batchId;

        int64_t batch_id2 = batch_id % N12;
        batch_id2 = batch_id2 != 0 ? batch_id2 : N12;
        batch_id = (Ceil<int64_t>(batch_id, N12) - 1) * N1 + N11 * k + batch_id2;

        if (p + q <= m) {
            if (n >= m) {
                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinateInfo.batchId = batch_id;
                } else {
                    y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;

                } else {
                    x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                }
                return;
            }
        } else {
            if (p + q <= n) {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                } else if (y < x - p + 1 && y + p + q - 1 <= n) {
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s2Idx = y + p + q - 1;
                } else {
                    coordinateInfo.batchId = -1;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    return;
                }
            }
        }
    }
    coordinateInfo.batchId = -1;
    return;
}
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_DETER_H_