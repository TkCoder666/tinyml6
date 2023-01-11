# TinyML 项目：在树莓派 4b 上运用多种方式对 CNN 进行推理过程加速

通过远程方式连接树莓派（[记录](https://gist.github.com/LogCreative/3b6af209d3fd6309cdfac02ed98f8789)），也可以通过虚拟机模拟（[记录](https://gist.github.com/LogCreative/f87d968d91cf554ccf48d3b3f7fd7987)）。

## 剪枝

参照教程(https://www.yuque.com/mnn/cn/fpy0dw)实现了MNN的三种剪枝方法，可运行下列命令进行剪枝。

```bash
python Pruner/SIMDOCPruner/mnn_prune.py 
```

测试了Pruning Filter in Filter的代码实现，具体代码可参考论文提供仓库(https://github.com/fxmeng/Pruning-Filter-in-Filter)

## 量化

首先按照教程(https://www.yuque.com/mnn/cn/vg3to5)安装MNN的python和cpp版本，并且安装相应的库。
接着直接用以下的命令进行训练量化：

```bash
python Quantizer/LSQQuantizer/LSQquan.py #启动LSQ量化
```

## TVM

首先需要在宿主机上安装 TVM（[记录](https://gist.github.com/LogCreative/8b8f0d956756cf710c01185eacc05d27)），在树莓派上安装 TVM Runtime，并通过 RPC 连接（[记录](https://gist.github.com/LogCreative/75eb8f87fb1d2ce227aa638216643776)）。

在宿主机上测试、微调：

```bash
python tvmc_host.py   # 编译优化与微调
python tvm_host.py    # 仅编译优化
```

在树莓派上测试、微调：

```bash
python tvmc_rasp.py   # 编译优化与微调
python tvm_rasp.py    # 仅编译优化
```
