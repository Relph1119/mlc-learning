# 《机器学习编译》的学习笔记

《机器学习编译》是陈天奇老师的MLC课程，主要包括机器学习编译、张量函数抽象、端到端模型整合、自动化程序优化、机器学习框架整合、自定义硬件后端、自动张量化、计算图优化、部署模型到服务环境、部署模型到边缘设备。基于TVMScript包，学习机器学习编译相关原理，主要围绕着TensorIR介绍。

《机器学习编译》课程主页：https://mlc.ai/summer22-zh/

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
codes------------------------------------------课程代码
|   +---ch02-------------------------------------第2章课程练习代码
|   |   +---lnumpy_mm_relu.py----------------------全连接层的张量函数
|   |   +---lnumpy_mm_relu_v2.py-------------------全连接层的张量函数V2
|   |   +---tensor_expression.py-------------------用张量表达式生成TensorIR
|   |   +---tensor_program_abstraction.py----------变换张量函数
|   +---ch04-------------------------------------第4章课程练习代码
|   |   +---end2end_lnumpy.py----------------------端到端模型的底层Numpy实现
|   |   +---end2end_numpy.py-----------------------端到端模型的numpy实现
|   |   +---end2end_tvmscript.py-------------------在TVMScript中构建端到端IRModule
|   |   +---mix_tensorir_lib.py--------------------将IRModule和运行时注册的函数混合执行
|   |   +---register_func.py-----------------------注册运行时函数
|   +---ch05-------------------------------------第5章课程练习代码
|   |   +---auto_schedule.py-----------------------单个元张量函数的自动调度
|   |   +---end2end_auto_optim_tensor_func.py------在端到端模型中使用自动搜索和自动调优
|   |   +---random_transform.py--------------------单个元张量函数的随机调度变换
|   |   +---random_transform_search.py-------------单个元张量函数的随机变换搜索
|   |   +---transform_signle_tensor_func.py--------变换单个元张量函数
|   +---ch06-------------------------------------第6章课程练习代码
|   |   +---blockbuilder_create_irmodule.py--------用BlockBuilder创建IRModule
|   |   +---fashion_mnist_example.py---------------FashionMNIST例子
|   |   +---pytorch_import_model.py----------------从PyTorch导入模型转成IRModule
|   |   +---tensor_expression_create_tensorir.py---Tensor表达式创建TensorIR
requirements.txt-------------------------------运行环境依赖包
</pre>