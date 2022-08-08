#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: gpu.py
@time: 2022/8/8 9:36
@project: mlc-learning
@desc: 滑动窗口求和(Window Sum Example)
"""

# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MyModuleWindowSum:
    @T.prim_func
    def main(A: T.Buffer[(1027,), "float32"],
             B: T.Buffer[(1024,), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi] + A[vi + 1] + A[vi + 2]


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MyModuleWindowSum)
    nthread = 128
    block_C = sch.get_block("C")
    i, = sch.get_loops(block=block_C)
    i0, i1 = sch.split(i, [None, nthread])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

    # 使用共享内存
    A_shared = sch.cache_read(block_C, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(A_shared, i1)

    # 线程绑定
    ax = sch.get_loops(A_shared)[-1]
    ax0, ax1 = sch.split(ax, [None, nthread])
    sch.bind(ax1, "threadIdx.x")
    # print(sch.mod.script())

    # 查看cuda的程序
    rt_mod = tvm.build(sch.mod, target="cuda")
    print(rt_mod.imported_modules[0].get_source())