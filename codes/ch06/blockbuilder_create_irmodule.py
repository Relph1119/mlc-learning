#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: blockbuilder_create_irmodule.py
@time: 2022/7/31 13:12
@project: mlc-learning
@desc: 用BlockBuilder创建IRModule
"""

from tvm import relax
from tvm import te


def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")


def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")


if __name__ == '__main__':
    A = relax.Var("A", (128, 128), relax.DynTensorType(2, "float32"))
    B = relax.Var("B", (128, 128), relax.DynTensorType(2, "float32"))

    bb = relax.BlockBuilder()

    with bb.function("main"):
        with bb.dataflow():
            C = bb.emit_te(te_matmul, A, B)
            D = bb.emit_te(te_relu, C)
            # 最后指定整合
            R = bb.emit_output(D)
        bb.emit_func_output(R, params=[A, B])

    MyModule = bb.get()
    print(MyModule.script())
