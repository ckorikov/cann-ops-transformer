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
import platform
import torch_npu
from setuptools import Extension
from torch_npu.utils.cpp_extension import TorchExtension

CANN_HOME = os.environ['ASCEND_TOOLKIT_HOME']
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
PLATFORM_ARCH = platform.machine() + "-linux"


def ascendc_extension(
    name,
    sources,
    extra_include_dirs,
    extra_library_dirs,
    extra_libraries
):
    kwargs = {}

    include_dirs = [
        os.path.join(CANN_HOME, PLATFORM_ARCH, 'include'),
        os.path.join(CANN_HOME, 'tools', 'tikcpp', 'tikcfw'),  # for `kernel tiling.h`
        os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'),
    ]
    include_dirs.extend(extra_include_dirs)
    include_dirs.extend(TorchExtension.include_paths())
    kwargs['include_dirs'] = include_dirs

    library_dirs = [
        os.path.join(CANN_HOME, PLATFORM_ARCH, 'lib64'),
        os.path.join(PYTORCH_NPU_INSTALL_PATH, 'lib'),
    ]
    library_dirs.extend(extra_library_dirs)
    library_dirs.extend(TorchExtension.library_paths())
    kwargs['library_dirs'] = library_dirs

    libraries = [
        'c10', 'torch', 'torch_cpu', 'torch_npu', 'torch_python', 'ascendcl', 'tiling_api'
    ]
    libraries.extend(extra_libraries)
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, **kwargs)