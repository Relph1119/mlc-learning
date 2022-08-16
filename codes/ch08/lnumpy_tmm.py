#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lnumpy_tmm.py
@time: 2022/8/16 9:44
@project: mlc-learning
@desc: 矩阵乘法
"""

# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import numpy as np


def accel_fill_zero(C):
    C[:] = 0


def accel_tmm_add(C, A, B):
    C[:] += A @ B.T


def accel_dma_copy(reg, dram):
    reg[:] = dram[:]


def lnumpy_tmm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # a special accumulator memory
    C_accumulator = np.empty((16, 16), dtype="float32")
    A_reg = np.empty((16, 16), dtype="float32")
    B_reg = np.empty((16, 16), dtype="float32")

    for i in range(64):
        for j in range(64):
            accel_fill_zero(C_accumulator[:, :])
            for k in range(64):
                accel_dma_copy(A_reg[:], A[i * 16: i * 16 + 16, k * 16: k * 16 + 16])
                accel_dma_copy(B_reg[:], B[j * 16: j * 16 + 16, k * 16: k * 16 + 16])
                accel_tmm_add(C_accumulator[:, :], A_reg, B_reg)
            accel_dma_copy(C[i * 16: i * 16 + 16, j * 16: j * 16 + 16], C_accumulator[:, :])


if __name__ == '__main__':
    dtype = "float32"
    a_np = np.random.rand(1024, 1024).astype(dtype)
    b_np = np.random.rand(1024, 1024).astype(dtype)
    c_tmm = a_np @ b_np.T

    c_np = np.empty((1024, 1024), dtype="float32")
    lnumpy_tmm(a_np, b_np, c_np)
    np.testing.assert_allclose(c_np, c_tmm, rtol=1e-5)
