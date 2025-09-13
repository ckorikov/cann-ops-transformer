#!/bin/bash
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# Open source rectification, kernel source code migrated to the op_kernel directory, equipped with operator kernel compilation script, 
# and first searched for operator files in the op_kernel directory.

DYNAMIC_PY_FILE=$1

GET_KERNEL_SOURCE_FUNCTION='def get_kernel_source(src_file, dir_snake, dir_ex):'
NEW_CODE='def get_kernel_source(src_file, dir_snake, dir_ex):\n    src = os.path.join(PYF_PATH, "op_kernel", src_file)\n    if os.path.exists(src):\n        return src'
sed -i "s/${GET_KERNEL_SOURCE_FUNCTION}/${NEW_CODE}/g" ${DYNAMIC_PY_FILE}