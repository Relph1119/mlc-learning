#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: matmul_tensor_intrin.py
@time: 2022/8/16 10:09
@project: mlc-learning
@desc: 矩阵乘法的张量化
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


@T.prim_func
def tmm16_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, scope="global.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def tmm16_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    sa = T.var("int32")
    sb = T.var("int32")
    sc = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, strides=[sa, 1], scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, strides=[sb, 1], scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, strides=[sc, 1], scope="global.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.call_extern(
                "tmm16",
                C.access_ptr("w"),
                A.access_ptr("r"),
                B.access_ptr("r"),
                sa,
                sb,
                sc,
                dtype="int32",
            )
        )


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MatmulModule)

    i, j, k = sch.get_loops("matmul")
    i, ii = sch.split(i, factors=[None, 16])
    j, ji = sch.split(j, factors=[None, 16])
    k, ki = sch.split(k, factors=[None, 16])
    sch.reorder(i, j, k, ii, ji, ki)
    block_mm = sch.blockize(ii)

    # 加入寄存器
    A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
    B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
    sch.compute_at(A_reg, k)
    sch.compute_at(B_reg, k)

    write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
    sch.reverse_compute_at(write_back_block, j)

    tvm.tir.TensorIntrin.register("tmm16", tmm16_desc, tmm16_impl)
    sch.decompose_reduction(block_mm, k)
    # sch.mod.show()

    sch.tensorize(block_mm, "tmm16")
    sch.mod.show()
