#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tensor_expression.py
@time: 2022/7/5 20:33
@project: mlc-learning
@desc: 用张量表达式生成TensorIR
"""
import tvm
from tvm import te

A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")

te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
print(MyModuleFromTE.script())
