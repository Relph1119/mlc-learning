#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: auto_schedule.py
@time: 2022/7/26 19:10
@project: mlc-learning
@desc: 单个元张量函数的自动调度
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import os
import shutil

import tvm
from tvm.script import tir as T
import numpy as np
from tvm import meta_schedule as ms


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


def del_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


if __name__ == '__main__':
    dtype = "float32"
    a_np = np.random.rand(128, 128).astype(dtype)
    b_np = np.random.rand(128, 128).astype(dtype)
    c_mm = a_np @ b_np

    a_nd = tvm.nd.array(a_np)
    b_nd = tvm.nd.array(b_np)
    c_nd = tvm.nd.empty((128, 128), dtype="float32")

    # 删除tune_tmp下面的所有文件
    del_dir("./tune_tmp")

    sch_tuned = ms.tune_tir(
        mod=MyModule,
        target="llvm --num-cores=1",
        config=ms.TuneConfig(
            max_trials_global=64,
            num_trials_per_iter=64,
        ),
        work_dir="./tune_tmp",
        task_name="main",
    )

    print(sch_tuned.trace)

    lib = tvm.build(sch_tuned.mod, target="llvm")
    f_timer_after = lib.time_evaluator("main", tvm.cpu())
    print("\nTime cost of MyModule after tuning: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
