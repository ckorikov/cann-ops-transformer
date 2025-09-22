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
#ifndef ARCH35_CATLASS_BLOCK_BLOCK_UTILS_H
#define ARCH35_CATLASS_BLOCK_BLOCK_UTILS_H

#include "../utils/device_utils.h"
namespace WeightQuantBatchMatmulV2::Arch35::Catlass {
template <size_t I, class Tuple>
struct deduce_optional_input {
private:
    static constexpr size_t validIndex =
        (I < AscendC::Std::tuple_size_v<Tuple> - 1) ? I : AscendC::Std::tuple_size_v<Tuple> - 1;

public:
    using type = AscendC::Std::conditional_t<
        (I < AscendC::Std::tuple_size_v<Tuple>), typename AscendC::Std::tuple_element<validIndex, Tuple>::type, void>;
};

namespace detail {

template <typename T>
DEVICE constexpr auto GetArgValue(T const& arg) -> decltype(arg)
{
    return arg;
}

template <typename T, T Value>
DEVICE constexpr T GetArgValue(AscendC::Std::integral_constant<T, Value>)
{
    return Value;
}

template <typename TupleT, typename ArgsTuple, size_t... I>
DEVICE auto Crd2idxHelper(TupleT const& t, ArgsTuple const& argsTuple, AscendC::Std::index_sequence<I...>)
{
    return (
        ... + (GetArgValue<typename AscendC::Std::tuple_element<I, TupleT>::type>(AscendC::Std::get<I>(t)) *
               GetArgValue<typename AscendC::Std::tuple_element<I, ArgsTuple>::type>(AscendC::Std::get<I>(argsTuple))));
}
} // namespace detail
template <size_t I, class Tuple>
using deduce_optional_input_t = typename deduce_optional_input<I, Tuple>::type;

template <typename... TupleTypes, typename... Args>
DEVICE auto crd2idx(AscendC::Std::tuple<TupleTypes...> const& t, Args... args)
{
    static_assert(sizeof...(Args) == sizeof...(TupleTypes), "The number of arguments must match the tuple size");

    auto argsTuple = AscendC::Std::make_tuple(args...);
    return detail::Crd2idxHelper(t, argsTuple, AscendC::Std::make_index_sequence<sizeof...(TupleTypes)>{});
}

} // namespace WeightQuantBatchMatmulV2::Arch35::Catlass
#endif