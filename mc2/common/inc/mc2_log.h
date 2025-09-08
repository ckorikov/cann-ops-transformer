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

/*!
 * \file mc2_log.h
 * \brief
 */

#ifndef MC2_LOG_H
#define MC2_LOG_H
#include <stdarg.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "base/err_msg.h"
#include "error_manager/error_manager.h"
#include "log/log.h"
#include "securec.h"
#include "tiling/mc2_tiling_struct.h"

template <typename T>
std::string ConcatString(const T &arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(T arg, Ts... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

namespace Mc2Log {
void PrintMMV3TilingData(const std::string &opName,
                         optiling::MC2MatmulV3TilingData &tiling);
void PrintTCubeTilingData(const std::string &opName,
                          optiling::TCubeTiling &tiling);
void PrintRCSTilingData(const std::string &opName,
                        optiling::RCSTiling &rcsTiling);
void PrintMc2MsgData(const std::string &opName, optiling::Mc2Msg &msg);
void PrintTileL2TilingData(const std::string &opName,
                           optiling::TileL2Tiling &tileL2Tiling);
}  // namespace Mc2Log

struct ErrorResult {
  operator bool() const { return false; }
  operator ge::graphStatus() const { return ge::GRAPH_PARAM_INVALID; }
  template <typename T>
  operator std::unique_ptr<T>() const {
    return nullptr;
  }
  template <typename T>
  operator std::shared_ptr<T>() const {
    return nullptr;
  }
  template <typename T>
  operator T *() const {
    return nullptr;
  }
  template <typename T>
  operator std::vector<std::shared_ptr<T>>() const {
    return {};
  }
  template <typename T>
  operator std::vector<T>() const {
    return {};
  }
  operator std::string() const { return ""; }
  template <typename T>
  operator T() const {
    return T();
  }
};

inline std::vector<char> CreateErrorMsg(const char *format, ...) {
  va_list args;
  va_start(args, format);
  va_list args_copy;
  va_copy(args_copy, args);
  const size_t len =
      static_cast<size_t>(vsnprintf(nullptr, 0, format, args_copy));
  va_end(args_copy);
  std::vector<char> msg(len + 1U, '\0');
  const auto ret = vsnprintf_s(msg.data(), len + 1U, len, format, args);
  va_end(args);
  return (ret > 0) ? msg : std::vector<char>{};
}

inline std::vector<char> CreateErrorMsg() { return {}; }

inline const char *get_cstr(const std::string &str) { return str.c_str(); }

#if !defined(__ANDROID__) && !defined(ANDROID)
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) \
  if (__builtin_expect(COND, 0)) {       \
    LOG_FUNC;                            \
    EXPR;                                \
  }

#define OPS_LOG_E(opName, ...) \
  D_OP_LOGE(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_I(opName, ...) \
  D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_D(opName, ...) \
  D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)              \
  do {                                                            \
    D_OP_LOGE(op_name, err_msg, ##__VA_ARGS__);                   \
    REPORT_INNER_ERR_MSG("E69999", "op[%s], " err_msg,            \
                         get_cstr(Ops::Base::GetOpInfo(op_name)), \
                         ##__VA_ARGS__);                          \
  } while (0)
#define GE_ASSERT(exp, ...)                                     \
  do {                                                          \
    if (!(exp)) {                                               \
      auto msg = CreateErrorMsg(__VA_ARGS__);                   \
      if (msg.empty()) {                                        \
        REPORT_INNER_ERROR("E19999", "Assert %S failed", #exp); \
        return ge::FAILED;                                      \
      } else {                                                  \
        REPORT_INNER_ERROR("E19999", "%S", msg.data());         \
        return ge::FAILED;                                      \
      }                                                         \
      return ::ErrorResult();                                   \
    }                                                           \
  } while (false)

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                         \
  if ((ptr) == nullptr) {                                                 \
    const char *name = ((context)->GetNodeName() == nullptr)              \
                           ? "nil"                                        \
                           : (context)->GetNodeName();                    \
    OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                 \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr); \
    return ge::GRAPH_FAILED;                                              \
  }
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                \
  if ((ptr) == nullptr) {                                                 \
    const char *name = ((context)->GetNodeName() == nullptr)              \
                           ? "nil"                                        \
                           : (context)->GetNodeName();                    \
    OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                 \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr); \
    return ret;                                                           \
  }
}  // namespace ops

#define GE_ASSERT_SUCCESS(v, ...) GE_ASSERT(((v) == ge::SUCCESS), __VA_ARGS__)
#define GE_ASSERT_NOTNULL(v, ...) GE_ASSERT(((v) != nullptr), __VA_ARGS__)
#define GE_ASSERT_GRAPH_SUCCESS(v, ...) GE_ASSERT(((v) == 0), __VA_ARGS__)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                  \
  do {                                                                         \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg));                  \
    std::string errorStr = "E89999";                                           \
    REPORT_INNER_ERR_MSG(errorStr.c_str(), "%s",                               \
                         ConcatString("op[", op_name, "],", err_msg).C_STR()); \
  } while (0)

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILING(op_name, err_msg, ...)     \
  do {                                                            \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);      \
    REPORT_INNER_ERR_MSG("E89999", "op[%s], " err_msg,            \
                         get_cstr(Ops::Base::GetOpInfo(op_name)), \
                         ##__VA_ARGS__);                          \
  } while (0)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#else
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR)
#define OPS_CHECK(COND, LOG_FUNC, EXPR)
#define OPS_LOG_E(opName, ...)
#define OPS_LOG_I(opName, ...)
#define OPS_LOG_D(opName, ...)
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, err_msg)
#define GE_ASSERT_SUCCESS(v, ...)
#define GE_ASSERT_NOTNULL(v, ...)
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILING(opName, err_msg, ...)
#define OP_TILING_CHECK(cond, log_func, expr)
}  // namespace optiling
#endif
#endif