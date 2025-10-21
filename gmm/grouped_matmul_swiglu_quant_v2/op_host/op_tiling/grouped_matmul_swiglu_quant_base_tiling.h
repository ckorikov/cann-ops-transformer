/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_GROUPED_MATMULSWIGLU_QUANT_BASE_TILING_H__
#define __OP_HOST_GROUPED_MATMULSWIGLU_QUANT_BASE_TILING_H__

#include "grouped_matmul_swiglu_quant_tiling.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {
namespace GroupedMatmulSwigluQuantV2Tiling {

class GroupedMatmulSwigluQuantV2BaseTiling : public GroupedMatmulSwigluQuantV2Tiling {
public:
    explicit GroupedMatmulSwigluQuantV2BaseTiling(gert::TilingContext* context) : GroupedMatmulSwigluQuantV2Tiling(context) {};

    ~GroupedMatmulSwigluQuantV2BaseTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    void FillTilingData() override;
    void PrintTilingData() override;
    void SetTilingKeyAndScheMode(void);
    ge::graphStatus ParseInputAndAttr();
    int64_t CalMaxRowInUbA8W4(const uint64_t ubSize, const uint64_t n);
    int64_t CalMaxRowInUb(const uint64_t ubSize, const uint64_t n);

private:
    GMMSwigluQuantV2TilingData tilingData_;
    uint32_t blockDim_;
    bool isA8W4MSD_;
    bool isSplitWorkSpace_;
    uint32_t groupNum_;
    int64_t k_;
    int64_t m_;
    int64_t n_;
    uint32_t maxProcessRowNum_;
    int64_t quantGroupNum_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint64_t workspaceSize_;
    int64_t mLimit_;
    int64_t usrWorkspaceLimut_;
};

}
}
#endif // __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_BASE_TILING_H__
