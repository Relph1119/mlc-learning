# 《机器学习编译》的学习笔记

《机器学习编译》是陈天奇老师的MLC课程，主要包括机器学习编译、张量函数抽象、端到端模型整合、自动化程序优化、机器学习框架整合、自定义硬件后端、自动张量化、计算图优化、部署模型到服务环境、部署模型到边缘设备。基于TVMScript包，学习机器学习编译相关原理，主要围绕TensorIR介绍。

《机器学习编译》课程主页：https://mlc.ai/summer22-zh/

资料整理：https://share.weiyun.com/hrgu6AzE

个人笔记部分主要包括：
1. 关键代码注释
2. 分开的练习代码，不采用notebook方式
3. 整理相关依赖包

## 运行环境配置
### Python版本
Python 3.7.9 Windows环境

### 运行环境配置
安装相关的依赖包
```shell
pip install -r requirements.txt
```

## 项目结构
<pre>
codes----------------------------------------------课程代码
|   +---ch02-----------------------------------------第2章 张量程序抽象
|   |   +---lnumpy_mm_relu.py--------------------------全连接层的张量函数
|   |   +---lnumpy_mm_relu_v2.py-----------------------全连接层的张量函数V2
|   |   +---tensor_expression.py-----------------------用张量表达式生成TensorIR
|   |   +---tensor_program_abstraction.py--------------变换张量函数
|   +---ch04-----------------------------------------第4章 端到端模型整合
|   |   +---end2end_lnumpy.py--------------------------端到端模型的底层Numpy实现
|   |   +---end2end_numpy.py---------------------------端到端模型的numpy实现
|   |   +---end2end_tvmscript.py-----------------------在TVMScript中构建端到端IRModule
|   |   +---mix_tensorir_lib.py------------------------将IRModule和运行时注册的函数混合执行
|   |   +---register_func.py---------------------------注册运行时函数
|   +---ch05-----------------------------------------第5章 自动化程序优化
|   |   +---auto_schedule.py---------------------------单个元张量函数的自动调度
|   |   +---end2end_auto_optim_tensor_func.py----------在端到端模型中使用自动搜索和自动调优
|   |   +---random_transform.py------------------------单个元张量函数的随机调度变换
|   |   +---random_transform_search.py-----------------单个元张量函数的随机变换搜索
|   |   +---transform_signle_tensor_func.py------------变换单个元张量函数
|   +---ch06-----------------------------------------第6章 与机器学习框架的整合
|   |   +---blockbuilder_create_irmodule.py------------用BlockBuilder创建IRModule
|   |   +---fashion_mnist_example.py-------------------FashionMNIST例子
|   |   +---pytorch_import_model.py--------------------从PyTorch导入模型转成IRModule
|   |   +---tensor_expression_create_tensorir.py-------Tensor表达式创建TensorIR
|   +---ch07-----------------------------------------第7章 GPU 硬件加速 1
|   |   +---matrix_multiplication.py-------------------优化矩阵乘法
|   |   +---shared_memory_blocking.py------------------共享内存块
|   |   +---tensor_add_with_gpu.py---------------------向量加法的GPU加速
|   |   +---window_sum.py------------------------------滑动窗口求和(Window Sum Example)
|   +---ch08-----------------------------------------第8章 GPU 硬件加速 2
|   |   +---lnumpy_tmm.py------------------------------矩阵乘法
|   |   +---matmul_block.py----------------------------带有张量化计算的block
|   |   +---matmul_blockization.py---------------------矩阵乘法的Blockization
|   |   +---matmul_tensor_intrin.py--------------------矩阵乘法的张量化
|   +---ch09-----------------------------------------第9章 计算图优化：算子融合和内存优化
|   |   +---fasionmnist_computational_graph_optim.py---基于Fasionmnist数据集的计算图优化
|   |   +---fasionmnist_mlp_fused.py-------------------基于fasionmnist融合Linear和ReLU算子
|   |   +---pattern_match_modify.py--------------------模式匹配和改写（访问者模式）
|   |   +---reflect_tensor_ir_calls.py-----------------映射到TensorIR Calls
requirements.txt-------------------------------------运行环境依赖包
</pre>

## 思考

### 1 TVM到底有什么用？
TVM的优势：  
1. 允许用户自定义torch/tensorflow中不存在的算子并生成高效实现
2. 使用自动调度器，平台相关的（arm/AMD GPU/NVIDIA GPU/intel cpu）性能调优
3. 对于部署，生成最小化的库，而不是需要依赖一个庞大的第三方库（例如cudnn）
4. 可以做计算图层面的优化（算子融合等等），可以接受tf/pytorch导出的模型并做相应的计算图和算子调优，提高模型的推理性能（TensorRT等框架也可以做到相关事情）