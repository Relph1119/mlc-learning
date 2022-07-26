#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tensor_program_abstraction.py.py
@time: 2022/7/5 16:32
@project: mlc-learning
@desc: 变换张量函数
"""

import numpy as np
import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer[128, "float32"],
             B: T.Buffer[128, "float32"],
             C: T.Buffer[128, "float32"]):
        # extra annotations for the function
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in range(128):
            with T.block("C"):
                # declare a data parallel iterator on spatial domain
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MyModule)
    # 得到block C
    block_c = sch.get_block("C")
    # 获得一个循环
    i, = sch.get_loops(block_c)
    # 拆分变换
    i0, i1, i2 = sch.split(i, factors=[None, 4, 4])
    print(sch.mod.script())
    # 交换顺序
    sch.reorder(i2, i1)
    print(sch.mod.script())
    # 并行化
    sch.parallel(i0)
    print(sch.mod.script())

    # 执行程序
    rt_mod = tvm.build(sch.mod, target="llvm")
    func = rt_mod["main"]

    # 定义输入参数
    a = tvm.nd.array(np.arange(128, dtype="float32"))
    b = tvm.nd.array(np.ones(128, dtype="float32"))
    # 内存分配
    c = tvm.nd.empty((128,), dtype="float32")

    # 调用main函数
    func(a, b, c)
    print(c)
