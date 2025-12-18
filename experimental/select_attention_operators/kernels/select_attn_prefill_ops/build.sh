#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This script builds the operator and installs a python torch extension package 'select_attn_prefill_ops'

# build operator as shared lib (.so file)
bash ./compile.sh

# build torch extension
rm -rf build select_attn_prefill_ops.egg-info
pip uninstall -y select_attn_prefill_ops
pip install . --no-build-isolation