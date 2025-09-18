#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#/

set(CMAKE_LIBRARY_OUTPUT_DIRECTOR ${PROJECT_SOURCE_DIR}/build)

file(GLOB_RECURSE OP_API_FILES "*.cpp")

add_library(opapi SHARED ${OP_API_FILES})

target_compile_definitions(opapi PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    LOG_CPP
)

target_include_directories(opapi PRIVATE
    ${OPAPI_INCLUDE}
)