#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: pattern_match_modify.py
@time: 2022/8/22 14:10
@project: mlc-learning
@desc: 模式匹配和改写（访问者模式）
"""
# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations

import tvm
from torch import Tensor
from tvm import relax
from tvm.script import relax as R


@tvm.script.ir_module
class MyModule:
    @R.function
    def main(x: Tensor((3, 4), "float32"), y: Tensor((3, 4), "float32")):
        with relax.dataflow():
            lv0 = relax.multiply(x, y)
            gv0 = relax.add(lv0, y)
            relax.output(gv0)
        return gv0


@relax.expr_functor.mutator
class EwiseFMARewriter(relax.PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        add_op = tvm.ir.Op.get("relax.add")
        multiply_op = tvm.ir.Op.get("relax.multiply")
        ewise_fma_op = tvm.ir.Op.get("relax.ewise_fma")

        if call.op != add_op:
            return call

        value = self.lookup_binding(call.args[0])
        if not isinstance(value, relax.Call) or value.op != multiply_op:
            return call

        fma_call = relax.Call(
            ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
        )
        return fma_call


if __name__ == '__main__':
    updated_fn = EwiseFMARewriter().visit_expr(MyModule["main"])
    # 改写程序
    updated_fn.show()
    # 简化节点
    relax.analysis.remove_all_unused(updated_fn).show()
