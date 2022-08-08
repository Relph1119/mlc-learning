#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: shared_memory_blocking.py
@time: 2022/8/8 10:23
@project: mlc-learning
@desc: 共享内存块
"""

# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tir as T
from tvm import meta_schedule as ms


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


def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    # 向量化
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")


def blocking_with_shared(
        sch,
        tile_local_y,
        tile_local_x,
        tile_block_y,
        tile_block_x,
        tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)

    return sch


if __name__ == '__main__':
    sch = tvm.tir.Schedule(MyModuleMatmul)
    sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)

    rt_mod = tvm.build(sch.mod, target="cuda")
    dev = tvm.cuda(0)
    evaluator = rt_mod.time_evaluator("main", dev, number=10)

    A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

    num_flop = 2 * 1024 * 1024 * 1024
    evaluator = rt_mod.time_evaluator("main", dev, number=10)
    print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))

    # 自动化优化
    sch_tuned = ms.tune_tir(
        mod=MyModuleMatmul,
        target="nvidia/tesla-p100",
        config=ms.TuneConfig(
            max_trials_global=64,
            num_trials_per_iter=64,
        ),
        work_dir="./tune_tmp",
        task_name="main"
    )

    print(sch_tuned.mod.script())

    rt_mod = tvm.build(sch_tuned.mod, target="nvidia/tesla-p100")
    dev = tvm.cuda(0)
    evaluator = rt_mod.time_evaluator("main", dev, number=10)

    print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
