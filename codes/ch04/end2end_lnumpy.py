#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: end2end_lnumpy.py
@time: 2022/7/13 10:08
@project: mlc-learning
@desc:  端到端模型的底层Numpy实现
"""

import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def get_dataset():
    # 加载数据集
    test_data = torchvision.datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return test_loader, class_names


def plot_img(img, label, class_names):
    plt.figure()
    plt.imshow(img[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print("Class:", class_names[label[0]])


class Lnumpy:
    @staticmethod
    def lnumpy_linear0(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
        Y = np.empty((1, 128), dtype="float32")
        for i in range(1):
            for j in range(128):
                for k in range(784):
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

        for i in range(1):
            for j in range(128):
                Z[i, j] = Y[i, j] + B[j]

    @staticmethod
    def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
        for i in range(1):
            for j in range(128):
                Y[i, j] = np.maximum(X[i, j], 0)

    @staticmethod
    def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
        Y = np.empty((1, 10), dtype="float32")
        for i in range(1):
            for j in range(10):
                for k in range(128):
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

        for i in range(1):
            for j in range(10):
                Z[i, j] = Y[i, j] + B[j]


def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    Lnumpy.lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    Lnumpy.lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    Lnumpy.lnumpy_linear1(lv1, w1, b1, out)
    return out


def lnumpy_call_tir(prim_func, inputs, shape, dtype):
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res


def lnumpy_mlp_with_call_tir(data, w0, b0, w1, b1):
    lv0 = lnumpy_call_tir(Lnumpy.lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32")
    lv1 = lnumpy_call_tir(Lnumpy.lnumpy_relu0, (lv0,), (1, 128), dtype="float32")
    out = lnumpy_call_tir(Lnumpy.lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32")
    return out

if __name__ == '__main__':
    test_loader, class_names = get_dataset()

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()
    plot_img(img, label, class_names)

    mlp_params = pkl.load(open("../model/fasionmnist_mlp_params.pkl", "rb"))
    # result = lnumpy_mlp(
    #     img.reshape(1, 784),
    #     mlp_params["w0"],
    #     mlp_params["b0"],
    #     mlp_params["w1"],
    #     mlp_params["b1"])

    result = lnumpy_mlp_with_call_tir(
        img.reshape(1, 784),
        mlp_params["w0"],
        mlp_params["b0"],
        mlp_params["w1"],
        mlp_params["b1"])

    print(result)
    pred_kind = result.argmax(axis=1)
    print(pred_kind)
    print("Low-level Numpy MLP Prediction:", class_names[pred_kind[0]])
