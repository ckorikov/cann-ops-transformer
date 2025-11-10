#!/usr/bin/python3
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from ascendc_extension import ascendc_extension
CURRENT_DIR = os.path.dirname(__file__)

PACKAGE_NAME = 'rope_matrix'  # pip package name
setup(
    name=PACKAGE_NAME,
    ext_modules=[
        ascendc_extension(
            name=PACKAGE_NAME,
            sources=['torch_interface.cpp', "op_host/rope_matrix_tiling.cpp", "op_host/rope_matrix_tiling.h"],
            extra_include_dirs=[
                os.path.join(CURRENT_DIR, "build_kernel", "include", "rope_matrix_custom_kernel"),
                os.path.join(CURRENT_DIR, "inc")
            ],
            extra_library_dirs=[
                os.path.join(CURRENT_DIR, "build_kernel", "lib")
            ],
            extra_libraries=['rope_matrix_custom_kernel']
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
