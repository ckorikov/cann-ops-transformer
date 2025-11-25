#!/usr/bin/fish
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set REAL_SHELL_PATH (realpath (command -v $argv[0]))
set MULTI_VERSION $argv[1]
set CANN_PATH (cd (dirname $REAL_SHELL_PATH)/../../ && pwd)
if test -d "$CANN_PATH/opp" -a test -d "$CANN_PATH/../latest"
    set INSATLL_PATH (cd (dirname $REAL_SHELL_PATH)/../../../ && pwd)
    if test -L "$INSATLL_PATH/latest/opp"
        set _ASCEND_OPP_PATH (cd $CANN_PATH/opp && pwd)
        if test "$MULTI_VERSION" = "multi_version"
            set _ASCEND_OPP_PATH (cd $INSATLL_PATH/latest/opp && pwd)
        end
    end
elseif test -d "$CANN_PATH/opp"
    set _ASCEND_OPP_PATH (cd $CANN_PATH/opp && pwd)
end

set -x ASCEND_OPP_PATH $_ASCEND_OPP_PATH

pylib_path="${_ASCEND_OPP_PATH}/python/site-packages/"
if test -d ${pylib_path}
    set -x PYTHONPATH $PYTHONPATH:$pylib_path
end

