#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lnumpy_mm_relu.py
@time: 2022/7/5 16:32
@project: mlc-learning
@desc: 全连接层的张量函数
"""

import numpy as np
import tvm
from tvm.script import tir as T


def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]

    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
             B: T.Buffer[(128, 128), "float32"],
             C: T.Buffer[(128, 128), "float32"]):
        # extra annotations for the function
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


@tvm.script.ir_module
class MyModuleWithTwoFunctions:
    @T.prim_func
    def mm(A: T.Buffer[(128, 128), "float32"],
           B: T.Buffer[(128, 128), "float32"],
           Y: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "mm", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

    @T.prim_func
    def relu(A: T.Buffer[(128, 128), "float32"],
             B: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))


if __name__ == '__main__':
    dtype = "float32"
    a_np = np.random.rand(128, 128).astype(dtype)
    b_np = np.random.rand(128, 128).astype(dtype)
    c_mm_relu = np.maximum(a_np @ b_np, 0)

    # 定义输入参数
    a_nd = tvm.nd.array(a_np)
    b_nd = tvm.nd.array(b_np)
    # 内存分配
    c_nd = tvm.nd.empty((128, 128), dtype=dtype)
    rt_lib = tvm.build(MyModule, target="llvm")
    rt_lib["mm_relu"](a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)
