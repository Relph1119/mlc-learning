#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: fasionmnist_computational_graph_optim.py
@time: 2022/8/22 15:38
@project: mlc-learning
@desc: 基于Fasionmnist数据集的计算图优化
"""

from __future__ import annotations

import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tvm
from tvm import relax, topi
from tvm.ir.module import IRModule


def create_model(mlp_params):
    bb = relax.BlockBuilder()
    x = relax.Var("x", (1, 784), relax.DynTensorType(2, "float32"))
    w0 = relax.const(mlp_params["w0"], "float32")
    b0 = relax.const(mlp_params["b0"], "float32")
    w1 = relax.const(mlp_params["w1"], "float32")
    b1 = relax.const(mlp_params["b1"], "float32")

    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.dense(x, w0))
            lv1 = bb.emit(relax.op.add(lv0, b0))
            lv2 = bb.emit(relax.op.relu(lv1))
            lv3 = bb.emit(relax.op.dense(lv2, w1))
            lv4 = bb.emit(relax.op.add(lv3, b1))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    return bb.get()


@relax.expr_functor.mutator
class DenseAddFusor(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod
        # cache pre-defined ops
        self.add_op = tvm.ir.Op.get("relax.add")
        self.dense_op = tvm.ir.Op.get("relax.nn.dense")
        self.counter = 0

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            # avoid already fused primitive functions
            if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
                continue
            updated_func = self.visit_expr(func)
            # 由于mlc-ai-nightly的当前版本（0.9.dev1958+g26d18beea）缺少remove_all_unused()方法，暂时注释掉
            # updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op

        # pattern match dense => add
        # 识别dense和add算子
        if not match_call(call, self.add_op):
            return call

        value = self.lookup_binding(call.args[0])
        if value is None:
            return call

        if not match_call(value, self.dense_op):
            return call

        x = value.args[0]
        w = value.args[1]
        b = call.args[1]

        # construct a new fused primitive function
        # 创建函数参数
        param_x = relax.Var("x", x.shape_, x._checked_type_)
        param_w = relax.Var("w", w.shape_, w._checked_type_)
        param_b = relax.Var("b", b.shape_, b._checked_type_)

        bb = relax.BlockBuilder()

        # 记录创建子函数的次数
        fn_name = "fused_dense_add%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_b]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.nn.dense(param_x, param_w))
                gv = bb.emit_output(relax.op.add(lv0, param_b))
            bb.emit_func_output(gv)

        # Add Primitive attribute to the fused funtions
        # 标记子函数
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        # construct call into the fused function
        return relax.Call(global_var, [x, w, b], None, None)


@tvm.ir.transform.module_pass(opt_level=2, name="DeseAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""

    def transform_module(self, mod, ctx):
        return DenseAddFusor(mod).transform()


@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map) -> None:
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()


def map_dense(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.dense, x, w)


def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)


def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])


op_map = {
    "relax.nn.dense": map_dense,
    "relax.add": map_add,
    "relax.nn.relu": map_relu
}


@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""

    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()


def get_dataset():
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


def plot_image(img, label, class_names):
    plt.figure()
    plt.imshow(img[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    print("Class:", class_names[label[0]])


if __name__ == '__main__':
    # 加载模型
    mlp_params = pkl.load(open("../model/fasionmnist_mlp_params.pkl", "rb"))

    # 优化计算图
    MLPModel = create_model(mlp_params)
    MLPFused = FuseDenseAddPass()(MLPModel)
    MLPModelTIR = LowerToTensorIRPass()(MLPFused)
    MLPModelFinal = relax.transform.FuseTIR()(MLPModelTIR)

    test_loader, class_names = get_dataset()

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()

    plot_image(img, label, class_names)

    ex = relax.vm.build(MLPModelFinal, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    data_nd = tvm.nd.array(img.reshape(1, 784))

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MLPModule Prediction:", class_names[pred_kind[0]])
