#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: matrix_multiplication.py
@time: 2022/8/8 10:12
@project: mlc-learning
@desc: 优化矩阵乘法
"""

# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"],
             B: T.Buffer[(1024, 1024), "float32"],
             C: T.Buffer[(1024, 1024), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    # 创建一个写入局部块
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    # 累加循环
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    # 循环展开
    sch.unroll(k1)
    # 设置顺序
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MyModuleMatmul)
    # 8*8的矩阵乘法
    sch = blocking(sch, 8, 8, 8, 8, 4)
    # print(sch.mod.script())

    rt_mod = tvm.build(sch.mod, target="cuda")
    dev = tvm.cuda(0)
    A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

    num_flop = 2 * 1024 * 1024 * 1024
    evaluator = rt_mod.time_evaluator("main", dev, number=10)

    print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))