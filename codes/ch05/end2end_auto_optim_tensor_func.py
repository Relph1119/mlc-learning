#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: end2end_auto_optim_tensor_func.py
@time: 2022/7/26 19:30
@project: mlc-learning
@desc: 在端到端模型中使用自动搜索和自动调优
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import os
import pickle as pkl
import shutil

import numpy as np
import torch
import torchvision
import tvm
from torch import Tensor
from tvm.script import relax as R
from tvm.script.parser import relax
from tvm.script import tir as T
from tvm import meta_schedule as ms


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


@tvm.script.ir_module
class MyModuleMixture:
    @T.prim_func
    def linear0(X: T.Buffer[(1, 784), "float32"],
                W: T.Buffer[(128, 784), "float32"],
                B: T.Buffer[(128,), "float32"],
                Z: T.Buffer[(1, 128), "float32"]):
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]

        for i, j in T.grid(1, 128):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((128, 784), "float32"),
             b0: Tensor((128,), "float32"),
             w1: Tensor((10, 128), "float32"),
             b1: Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.call_tir(linear0, (x, w0, b0), (1, 128), dtype="float32")
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


def del_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


if __name__ == '__main__':
    test_loader, class_names = get_dataset()

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()

    mlp_params = pkl.load(open("../model/fasionmnist_mlp_params.pkl", "rb"))

    data_nd = tvm.nd.array(img.reshape(1, 784))
    nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

    MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)

    ex = relax.vm.build(MyModuleWithParams, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])

    # 打印运行时间
    ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)
    print("MyModuleWithParams time-cost: %g ms" % (ftimer(data_nd).mean * 1000))

    # 删除tune_tmp下面的所有文件
    del_dir("./tune_tmp")

    mod_linear = tvm.IRModule.from_expr(MyModuleMixture["linear0"].with_attr("global_symbol", "main"))
    sch_tuned_linear = ms.tune_tir(
        mod=mod_linear,
        target="llvm --num-cores=1",
        config=ms.TuneConfig(
            max_trials_global=64,
            num_trials_per_iter=64,
        ),
        work_dir="./tune_tmp",
        task_name="main",
    )

    # 在调优后用新函数替换原来的linear0
    MyModuleWithParams2 = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
    new_func = sch_tuned_linear.mod["main"].with_attr("global_symbol", "linear0")
    gv = MyModuleWithParams2.get_global_var("linear0")
    MyModuleWithParams2.update_func(gv, new_func)

    ex = relax.vm.build(MyModuleWithParams2, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithParams2 Prediction:", class_names[pred_kind[0]])

    # 打印运行时间
    ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=50)
    print("MyModuleWithParams2 time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
