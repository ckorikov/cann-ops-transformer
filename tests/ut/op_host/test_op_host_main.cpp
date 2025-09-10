/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <gtest/gtest.h>

#include "platform/platform_info.h"

using namespace std;

class OpTilingUtEnvironment : public testing::Environment {
  public:
    OpTilingUtEnvironment() {}
    virtual void SetUp() {
      cout << "Global Environment SetpUp." << endl;
      fe::OptionalInfos opti_compilation_infos_ge;
      opti_compilation_infos_ge.Init();
      opti_compilation_infos_ge.SetSocVersion("soc_version");
      fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(opti_compilation_infos_ge);
    }

    virtual void TearDown() {
      cout << "Global Environment TearDown" << endl;
    }
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc,argv);
  testing::AddGlobalTestEnvironment(new OpTilingUtEnvironment());
  return RUN_ALL_TESTS();
}
