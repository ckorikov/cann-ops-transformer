/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

/*!
 * \file ifa_flag_data.h
 * \brief
 */

#ifndef IFA_FLAG_DATA_H
#define IFA_FLAG_DATA_H

struct IFAFlagData {
  uint64_t tscmIdx : 2;  // query tscm que idx
  uint64_t tscmReuse : 1;
  uint64_t rsvd : 61;  // 保留
};

#endif  // IFA_FLAG_DATA_H
