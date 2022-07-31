#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: fashion_mnist_example.py
@time: 2022/7/31 13:43
@project: mlc-learning
@desc: FashionMNIST例子
"""

from torch import fx
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from tvm import topi
from tvm import relax
from tvm import te
import tvm


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


def plot_img(img, label, class_names):
    plt.figure()
    plt.imshow(img[0])
    plt.colorbar()
    plt.grid(False)
    # plt.show()
    print("Class:", class_names[label[0]])


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        return x


def map_param(param: nn.Parameter):
    ndim = len(param.data.shape)
    return relax.const(
        param.data.cpu().numpy(), relax.DynTensorType(ndim, "float32")
    )


def map_nn_linear(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    if nn_mod.bias is not None:
        b = map_param(nn_mod.bias)
    y = bb.emit_te(topi.nn.dense, x, w)
    return bb.emit_te(topi.add, y, b)


def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")


def map_relu(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    return bb.emit_te(te_relu, A)


def map_nn_relu(bb, node_map, node, nn_mod):
    return map_relu(bb, node_map, node)


def fetch_attr(fx_mod, target: str):
    """Helper function to fetch an attr"""
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, shape, relax.DynTensorType(len(shape), "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()


if __name__ == '__main__':
    test_loader, class_names = get_dataset()

    img, label = next(iter(test_loader))
    img = img.reshape(1, 28, 28).numpy()
    plot_img(img, label, class_names)

    import pickle as pkl

    mlp_model = MLP()

    mlp_params = pkl.load(open("../model/fasionmnist_mlp_params.pkl", "rb"))
    mlp_model.linear0.weight.data = torch.from_numpy(mlp_params["w0"])
    mlp_model.linear0.bias.data = torch.from_numpy(mlp_params["b0"])
    mlp_model.linear1.weight.data = torch.from_numpy(mlp_params["w1"])
    mlp_model.linear1.bias.data = torch.from_numpy(mlp_params["b1"])

    torch_res = mlp_model(torch.from_numpy(img.reshape(1, 784)))

    pred_kind = np.argmax(torch_res.detach().numpy(), axis=1)
    print("Torch Prediction:", class_names[pred_kind[0]])

    MLPModule = from_fx(
        fx.symbolic_trace(mlp_model),
        input_shapes=[(1, 784)],
        call_function_map={
        },
        call_module_map={
            torch.nn.Linear: map_nn_linear,
            torch.nn.ReLU: map_nn_relu,
        },
    )

    # print(MLPModule.script())
    ex = relax.vm.build(MLPModule, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    data_nd = tvm.nd.array(img.reshape(1, 784))

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MLPModule Prediction:", class_names[pred_kind[0]])
