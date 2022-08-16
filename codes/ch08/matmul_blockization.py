#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: matmul_blockization.py
@time: 2022/8/16 10:04
@project: mlc-learning
@desc: 矩阵乘法的Blockization
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import tvm
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
            A: T.Buffer[(1024, 1024), "float32"],
            B: T.Buffer[(1024, 1024), "float32"],
            C: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MatmulModule)
    i, j, k = sch.get_loops("matmul")
    i, ii = sch.split(i, factors=[None, 16])
    j, ji = sch.split(j, factors=[None, 16])
    k, ki = sch.split(k, factors=[None, 16])
    sch.reorder(i, j, k, ii, ji, ki)
    # sch.mod.show()

    block_mm = sch.blockize(ii)
    # sch.mod.show()

    # 引入特殊内存层级
    A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
    B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
    sch.compute_at(A_reg, k)
    sch.compute_at(B_reg, k)

    write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
    sch.reverse_compute_at(write_back_block, j)
    sch.mod.show()
