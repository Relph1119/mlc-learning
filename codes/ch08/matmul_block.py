#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: matmul_block.py
@time: 2022/8/16 9:53
@project: mlc-learning
@desc: 带有张量化计算的block
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import numpy as np
import tvm
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.script.ir_module
class MatmulBlockModule:
    @T.prim_func
    def main(
            A: T.Buffer[(1024, 1024), "float32"],
            B: T.Buffer[(1024, 1024), "float32"],
            C: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, j0, k0 in T.grid(64, 64, 64):
            with T.block("tmm-16x16"):
                vi0, vj0, vk0 = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.block("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0)

                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.block("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        C[vi0 * 16 + vi1, vj0 * 16 + vj1] += \
                            A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]


if __name__ == '__main__':
    # 显示MatmulBlockModule的结构
    # MatmulBlockModule.show()

    # 测试矩阵乘法的函数功能
    dtype = "float32"
    a_np = np.random.rand(1024, 1024).astype(dtype)
    b_np = np.random.rand(1024, 1024).astype(dtype)
    c_tmm = a_np @ b_np.T

    a_nd = tvm.nd.array(a_np)
    b_nd = tvm.nd.array(b_np)

    c_nd = tvm.nd.empty((1024, 1024), dtype="float32")

    lib = tvm.build(MatmulBlockModule, target="llvm")
    lib["main"](a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), c_tmm, rtol=1e-5)

    # 变换在张量化block周围的循环
    sch = tvm.tir.Schedule(MatmulBlockModule)

    block_mm = sch.get_block("tmm-16x16")
    i, j, k = sch.get_loops(block_mm)

    i0, i1 = sch.split(i, [None, 4])

    sch.reorder(i0, j, i1, k)
    sch.mod.show()