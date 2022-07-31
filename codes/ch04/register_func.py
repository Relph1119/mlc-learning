#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: register_func.py
@time: 2022/7/26 17:27
@project: mlc-learning
@desc: 注册运行时函数
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tvm
from torch import Tensor
from tvm.script import relax as R
from tvm import relax


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


@tvm.script.ir_module
class MyModuleWithExternCall:
    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((128, 784), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((10, 128), "float32"),
             b1: Tensor((10,), "float32")):
        # block 0
        with R.dataflow():
            lv0 = R.call_tir("env.linear", (x, w0, b0), (1, 128), dtype="float32")
            lv1 = R.call_tir("env.relu", (lv0,), (1, 128), dtype="float32")
            out = R.call_tir("env.linear", (lv1, w1, b1), (1, 10), dtype="float32")
            R.output(out)
        return out


@tvm.register_func("env.linear", override=True)
def torch_linear(x: tvm.nd.NDArray,
                 w: tvm.nd.NDArray,
                 b: tvm.nd.NDArray,
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)


@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray,
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)


if __name__ == '__main__':
    test_loader, class_names = get_dataset()

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()
    plot_img(img, label, class_names)

    mlp_params = pkl.load(open("../model/fasionmnist_mlp_params.pkl", "rb"))

    data_nd = tvm.nd.array(img.reshape(1, 784))
    nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

    ex = relax.vm.build(MyModuleWithExternCall, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd,
                        nd_params["w0"],
                        nd_params["b0"],
                        nd_params["w1"],
                        nd_params["b1"])

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])
