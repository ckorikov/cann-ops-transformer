
#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This script only compiles the AscendC operator and creates a shared library file in lib directory

# `bisheng` CLI: https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/BishengCompiler/atlas_bisheng_10_0003.html
# Adapted from: https://gitee.com/ascend/mstt/tree/master/sample/pytorch_adapter

# validate cann environment
source ../../scripts/num_cores_map.sh
source ../../scripts/check_cann.sh
if ! check_cann_environment false; then
    exit 1
fi

# to place the .so library there
mkdir -p lib
rm -f lib/libquest_prefill_metadata.so

# normal compilation
bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --cce-soc-version=$SOC_VERSION --cce-soc-core-type=VecCore \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -o lib/libquest_prefill_metadata.so quest_prefill_metadata.cpp