#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""函数基础库。"""

import operator
from functools import partial, wraps
from operator import itemgetter
from typing import Callable, Dict, Iterator, Optional, Tuple, TypeVar


A = TypeVar('A')


def groupbydict(objs: Iterator, keyfunc: Callable, valuefunc: Callable = None) -> Dict:
    """数据分组为字典。"""
    results = {}
    for obj in objs:
        group = results.setdefault(keyfunc(obj), [])
        if valuefunc is None:
            group.append(obj)
        else:
            group.append(valuefunc(obj))

    return results


def transpose(mat: Tuple[Tuple, ...]) -> Tuple[Tuple, ...]:
    """矩阵转置。"""
    if not mat:
        return mat

    width = len(mat[0])
    result = [[] for _ in range(width)]
    for row in mat:
        for col_idx in range(width):
            result[col_idx].append(row[col_idx])

    result = tuple(tuple(col) for col in result)
    return result


def constant(value: A) -> Callable[..., A]:
    """常量值。"""
    def constant_inner(*_args, **_kwargs) -> A:
        return value

    return constant_inner


def debug(value: A) -> A:
    """。试调"""
    breakpoint()  # pylint: disable=forgotten-debug-statement
    return value


def dispatch(*funcs):
    """分派应用。"""
    def dispatch_inner(*args, **kwargs) -> Iterator:
        return (func(*args, **kwargs) for func in funcs)

    return dispatch_inner


def cross(*funcs) -> Callable:
    """交错应用。"""
    def cross_func(data):
        return (func(x) for func, x in zip(funcs, data))
    return cross_func


def pipe(*funcs):
    """串联多个函数。"""
    def pipe_func(*args, **kargs):
        result = funcs[0](*args, **kargs)
        for func in funcs[1:]:
            result = func(result)
        return result

    return pipe_func


def partial_lift(func, *args, **kwargs):
    """部分化提升。

    提升函数的应用维度，提升时对函数做部分化。
    """
    return partial(map, partial(func, *args, **kwargs))


def decorate(preprocess: Optional[Callable],
             postprocess: Optional[Callable],
             func: Callable) -> Callable:
    """装饰函数。"""
    if preprocess and postprocess:
        return pipe(preprocess, func, postprocess)
    if preprocess:
        return pipe(preprocess, func)
    if postprocess:
        return pipe(func, postprocess)
    return func


def identity(value: A) -> A:
    """同一。"""
    return value


def invoke(func, *args, **kwargs):
    """调用。"""
    return func(*args, **kwargs)


def start_pipe(init, pipe_func):
    """启动管道。"""
    return invoke(pipe_func, init)


def side_effect(*funcs):
    """调用函数，产生副作用，但不影响管道结果。"""
    def side_effect_func(arg):
        for func in funcs:
            # 不保留结果
            func(arg)
        return arg

    return side_effect_func


def star_apply(func):
    """列表展开再应用。"""
    def star_apply_func(arg):
        return func(*arg)

    return star_apply_func


def with_statement(with_func: Callable, call_func: Callable) -> Iterator:
    """with语句。"""
    with with_func() as some:
        yield from call_func(some)


def select_fields(*fields) -> Callable:
    """选取字段。"""
    getters = (itemgetter(field) for field in fields)
    return dispatch(*getters)


def any_(*funcs) -> Callable:
    """高阶any。
    注意，any有短路效果。"""
    return pipe(
        dispatch(*funcs),
        any,
    )


def all_(*funcs) -> Callable:
    """高阶all。
    注意，all有短路效果。"""
    return pipe(
        dispatch(*funcs),
        all,
    )


def not_(func) -> Callable:
    """高阶not。"""
    return pipe(func, operator.not_)


def lazy_eval(func):
    """延迟求值。将一个函数转换为一个迭代器。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        yield value
    return wrapper


def expand_groupby(expand_func: Callable = tuple):
    """展开groupby。"""
    return pipe(partial(map, pipe(cross(identity, expand_func), tuple)))


def maybe(func):
    """可能为空。"""
    @wraps(func)
    def wrapper(some):
        if some is None:
            return None
        return func(some)
    return wrapper
