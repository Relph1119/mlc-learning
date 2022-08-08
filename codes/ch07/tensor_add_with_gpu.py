#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tensor_add_with_gpu.py
@time: 2022/8/8 9:24
@project: mlc-learning
@desc: 向量加法的GPU加速
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MyModuleVecAdd:
    @T.prim_func
    def main(A: T.Buffer[(1024,), "float32"],
             B: T.Buffer[(1024,), "float32"],
             C: T.Buffer[(1024,), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MyModuleVecAdd)
    block_C = sch.get_block("C")
    i, = sch.get_loops(block=block_C)
    i0, i1 = sch.split(i, [None, 128])

    # 映射到线程上
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")
    print(sch.mod.script())

    # 编译运行在GPU上
    rt_mod = tvm.build(sch.mod, target="cuda")

    A_np = np.random.uniform(size=(1024,)).astype("float32")
    B_np = np.random.uniform(size=(1024,)).astype("float32")
    A_nd = tvm.nd.array(A_np, tvm.cuda(0))
    B_nd = tvm.nd.array(B_np, tvm.cuda(0))
    C_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))

    rt_mod["main"](A_nd, B_nd, C_nd)
    print(A_nd)
    print(B_nd)
    print(C_nd)
