/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ARCH35_CATLASS_UTILS_DEVICE_UTILS_H
#define ARCH35_CATLASS_UTILS_DEVICE_UTILS_H

#define DEVICE __aicore__ inline

#if defined(__CCE_KT_TEST__)
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define X_LOG(format, ...)                                                                                  \
    do {                                                                                                    \
        std::string coreType = "";                                                                          \
        std::string blockId = "Block_";                                                                     \
        if (g_coreType == AscendC::AIC_TYPE) {                                                              \
            coreType = "AIC_";                                                                              \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                       \
            coreType = "AIV_";                                                                              \
        } else {                                                                                            \
            coreType = "MIX_";                                                                              \
        }                                                                                                   \
        coreType += std::to_string(sub_block_idx);                                                          \
        blockId += std::to_string(block_idx);                                                               \
        printf(                                                                                             \
            "[%s][%s][%s:%d][%s][%ld] " format "\n", blockId.c_str(), coreType.c_str(), FILENAME, __LINE__, \
            __FUNCTION__, (long)getpid(), ##__VA_ARGS__);                                                   \
    } while (0)

#else
#define X_LOG(format, ...)
#endif
#endif