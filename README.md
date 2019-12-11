# Constnet- based (forked) on the fixup repository.

Is Feature Diversity Necessary in Neural Network Initialization?

Example command:
python train.py --layers 16 --widen-factor 10 --fixup --batchnorm --lr 0.03 --name cifar10_40_layers -d 0:1 --droprate 0.15 --dataset cifar10

The code also supports LTH like pruning.

Source code description:
# Wide Residual Network with optional Fixup initialization

The code presents the implementation of Fixup as an option for standard Wide ResNet. When BatchNorm and Fixup are enabled simultaneously, Fixup initialization and the standard structure of the residual block are used.

Usage example:

```sh
python train.py --layers 40 --widen-factor 10 --batchnorm False --fixup True
```

# Acknowledgment
[Wide Residual Network](https://arxiv.org/abs/1605.07146) by Sergey Zagoruyko and Nikos Komodakis

[Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321) by Hongyi Zhang, Yann N. Dauphin, Tengyu Ma

Fixup implementation was originally [introduced by Andy Brock](https://github.com/ajbrock/BoilerPlate)

[WRN code](https://github.com/xternalz/WideResNet-pytorch) by xternalz
