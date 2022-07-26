#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: end2end_numpy.py
@time: 2022/7/26 12:25
@project: mlc-learning
@desc: 端到端模型的numpy实现
"""
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


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


def numpy_mlp(data, w0, b0, w1, b1):
    """
    模型的numpy实现
    """
    lv0 = data @ w0.T + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1.T + b1
    return lv2


if __name__ == '__main__':
    test_loader, class_names = get_dataset()

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()
    plot_img(img, label, class_names)

    mlp_params = pkl.load(open("../model/fasionmnist_mlp_params.pkl", "rb"))
    res = numpy_mlp(img.reshape(1, 784),
                    mlp_params["w0"],
                    mlp_params["b0"],
                    mlp_params["w1"],
                    mlp_params["b1"])
    print(res)
    pred_kind = res.argmax(axis=1)
    print(pred_kind)
    print("NumPy-MLP Prediction:", class_names[pred_kind[0]])
