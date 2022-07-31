#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tensor_expression_create_tensorir.py
@time: 2022/7/31 12:40
@project: mlc-learning
@desc: Tensor表达式创建TensorIR
"""

from tvm import te


def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")


def test_te_matmul():
    A = te.placeholder((128, 128), name="A", dtype="float32")
    B = te.placeholder((128, 128), name="B", dtype="float32")
    C = te_matmul(A, B)

    print(te.create_prim_func([A, B, C]).script())


def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")


def test_te_relu():
    X1 = te.placeholder((10,), name="X1", dtype="float32")
    Y1 = te_relu(X1)

    print(te.create_prim_func([X1, Y1]).script())


if __name__ == '__main__':
    test_te_relu()
