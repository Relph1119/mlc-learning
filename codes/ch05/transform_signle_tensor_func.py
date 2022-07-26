#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: transform_signle_tensor_func.py
@time: 2022/7/26 18:13
@project: mlc-learning
@desc: 变换单个元张量函数
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import tvm
from tvm.script import tir as T
import numpy as np


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer[(128, 128), "float32"],
             B: T.Buffer[(128, 128), "float32"],
             C: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def schedule_mm(sch: tvm.tir.Schedule, jfactor=4):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch


def schedule_mm_test(sch, print_script=False):
    sch = schedule_mm(sch)

    if print_script:
        print(sch.mod.script())

    lib = tvm.build(sch.mod, target="llvm")
    f_timer_after = lib.time_evaluator("main", tvm.cpu())
    print("Time cost of MyModule=>schedule_mm: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))


if __name__ == '__main__':
    dtype = "float32"
    a_np = np.random.rand(128, 128).astype(dtype)
    b_np = np.random.rand(128, 128).astype(dtype)
    c_mm = a_np @ b_np

    a_nd = tvm.nd.array(a_np)
    b_nd = tvm.nd.array(b_np)
    c_nd = tvm.nd.empty((128, 128), dtype="float32")

    lib = tvm.build(MyModule, target="llvm")
    f_timer_before = lib.time_evaluator("main", tvm.cpu())
    print("Time cost of MyModule: %.3f ms" % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))
    sch = tvm.tir.Schedule(MyModule)
    # 变换的历史轨迹
    # print(sch.trace)

    sch = tvm.tir.Schedule(MyModule)

    schedule_mm_test(sch, print_script=True)
