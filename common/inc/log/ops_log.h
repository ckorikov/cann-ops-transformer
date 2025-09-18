/**
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ops_log.h
 * \brief
 */

#pragma once

/* 基础日志 */
#define OPS_LOG_D(OPS_DESC, ...) 
#define OPS_LOG_I(OPS_DESC, ...) 
#define OPS_LOG_W(OPS_DESC, ...) 
#define OPS_LOG_E(OPS_DESC, ...) 
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...)
#define OPS_LOG_EVENT(OPS_DESC, ...) 

/* 全量日志
 * 输出超长日志, 若日志超长, 则会被分为多行输出 */
#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...) 
#define OPS_LOG_D_FULL(OPS_DESC, ...) 
#define OPS_LOG_I_FULL(OPS_DESC, ...) 
#define OPS_LOG_W_FULL(OPS_DESC, ...) 

/* 条件日志 */
#define OPS_LOG_D_IF(COND, OP_DESC, EXPR, ...) 
#define OPS_LOG_I_IF(COND, OP_DESC, EXPR, ...) 
#define OPS_LOG_W_IF(COND, OP_DESC, EXPR, ...) 
#define OPS_LOG_E_IF(COND, OP_DESC, EXPR, ...) 
#define OPS_LOG_EVENT_IF(COND, OP_DESC, EXPR, ...) 

#define OPS_CHECK(COND, LOG_FUNC, EXPR)                                                                                \
    if (COND) {                                                                                                        \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }
