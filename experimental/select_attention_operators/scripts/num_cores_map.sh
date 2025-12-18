#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

declare -A NumCoresMap
NumCoresMap["Ascend910A"]="32"
NumCoresMap["Ascend910B1"]="25"
NumCoresMap["Ascend910B2"]="24"
NumCoresMap["Ascend910B3"]="20"
NumCoresMap["Ascend910B4"]="20"

export NumCoresMap