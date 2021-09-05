## DSTAdam

The PyTorch implementation of DSTAdam algorithm in：'A decreasing scaling transition scheme from Adam to SGD'
[https://arxiv.org/abs/2106.06749](https://arxiv.org/abs/2106.06749)
The implementation is highly based on projects [AdaBound](https://github.com/Luolc/AdaBound) , [Adam](https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/adam.py) , [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar), thanks pretty work.
The test environment we passed is: PyTorch=1.7.0, python=3.7.10.

### Usage

Please directly download the [dstadam](https://github.com/kunzeng/DSTAdam/dstadam) folder and put it in your project, then

```python
from dstadam import DSTAdam 

...

optimizer = DSTAdam(model.parameters(), iters=required, coeff=1e-8, up_lr=1, low_lr=0.005)

...

#iters(int, required): iterations
#	 iters = math.ceil(train_size / batch_size) * epochs
#
#rho(float, optional: transition factor
#    rho = 10 ** (math.log(coeff,10) / iters)

#DSTAdam default value: coeff=1e-5, up_lr=1, low_lr=0.005

```

## Demos：CIFAR
We give the codes of training CIFAR10/100 using DSTAdam in this paper.

### Usage
For CIFAR-10
```python
python main.py    # equal to:  python main.py --cifar="cifar10" --optimizer="DSTAdam" --model="resnet18" --lr=0.001 --coeff=1e-8 --up_lr=5 --low_lr=0.005

```

For CIFAR-100
```python
python main.py --cifar=cifar100   # equal to:  python main.py --cifar="cifar10" --optimizer="DSTAdam" --model="resnet18" --lr=0.001 --coeff=1e-8 --up_lr=5 --low_lr=0.005

```



