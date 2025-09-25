#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import os
import sys
np.random.seed(2025)

class Object():
    def __init__(self, config):
        self.config = config
        self.batchSize = config["batchSize"]
        self.compressBlockSize = config["compressBlockSize"]
        self.compressStride = config["compressStride"]
        self.headNum = config["headNum"]
        self.headDim = config["headDim"]
        self.headNumDim = self.headNum * self.headDim

        if (config["seqIsRandom"]):
            self.actSeqLenOptional = np.random.randint(0, config["maxSeqLen"], [self.batchSize]).astype(np.int64)
        else:
            self.actSeqLenOptional = np.array([config["maxSeqLen"] for _ in range(self.batchSize)]).astype(np.int64)
        self.actSeqLenOptional = np.cumsum(self.actSeqLenOptional).astype(np.int64)

        if(config["inputIsRandom"]):
            self.input = np.random.uniform(0.0, 1.0, [self.actSeqLenOptional[-1], self.headNumDim]).astype(np.float16)
            self.weight = np.random.uniform(0.0, 1.0, [self.compressBlockSize, self.headNum]).astype(np.float16)
        else:
            self.input = np.ones((self.actSeqLenOptional[-1], self.headNumDim), dtype=np.float16)
            self.weight = np.ones((self.compressBlockSize, self.headNum), dtype=np.float16)     

    def save_data(self):
        with open("config.txt", "w") as f:
            for key, value in self.config.items():
                f.write(f"{key}:{value}\n")

        self.input.tofile("input.bin")
        self.weight.tofile("weight.bin")
        self.actSeqLenOptional.tofile("actSeqLenOptional.bin")

def gen_data():
    config = {
        "batchSize" : 48,
        "compressBlockSize": 32,
        "compressStride" : 32,
        "headNum" : 24,
        "headDim" : 192,
        "seqIsRandom": False,
        "inputIsRandom": False,
        "maxSeqLen": 100
    }
    obj = Object(config)
    obj.save_data()

if __name__ == "__main__":
    case_name = sys.argv[1]
    gen_data()