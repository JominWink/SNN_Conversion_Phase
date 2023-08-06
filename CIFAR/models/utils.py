import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

def ceil_ste(x):
    return (x.ceil() - x).detach() + x

class StraightThrough(nn.Module):
    """

    """
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


myfloor = GradFloor.apply

class AvgPoolConv(nn.Conv2d):
    """
    Converting the AvgPool layers to a convolution-wrapped module,
    so that this module can be identified in Spiking-refactor.
    """
    def __init__(self, kernel_size=2, stride=2, input_channel=64, padding=0, freeze_avg=True):
        super().__init__(input_channel, input_channel, kernel_size, padding=padding, stride=stride,
                         groups=input_channel, bias=False)
        # init the weight to make them equal to 1/k/k
        self.set_weight_to_avg()
        self.freeze = freeze_avg
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        self.set_weight_to_avg()
        x = super().forward(*inputs)
        return self.relu(x)

    def set_weight_to_avg(self):
        self.weight.data.fill_(1).div_(self.kernel_size[0] * self.kernel_size[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count

def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.item():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower() \
            or 'MyFloor_by_phaseII' in name.lower() or 'Myfloor' in name.lower():
        return True
    return False

class TCL(nn.Module):
    '''
        torch.clamp(input, min, max, out->none)
    '''
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([8.]), requires_grad=True)
    def forward(self, x):
        x = F.relu(x, inplace='True') #inplace=True 类似于C的地址传递(引用)
        x = self.up - x
        x = F.relu(x, inplace='True')
        x = self.up - x
        return x

class MyFloor(nn.Module):
    def __init__(self, up=8., t=8, alpha = 0.5):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        x = myfloor(x * self.t) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

class MyFloor_by_phaseII(nn.Module): #λ = 8, L = 16 up = λ 代替ReLU激活函数的QCFS自定义激活函数
    def __init__(self, up=8., t=8, alpha = 0.5):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True) #让网络在训练进行反向传播的时候自动计算它的梯度
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        u = F.relu(self.alpha)
        x = myfloor(x * self.t + u) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

def replace_activation_by_floor(model, t=8): # t = 16 递归实现替代ReLU激活函数
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                # print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor_by_phaseII(module.up.item(), t)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(8., t)

    return model

def set_threshold_by_Myfloor(model, module):
    modules_myfloor = []
    for n, m in model.named_modules():
        if isinstance(m, MyFloor):
            modules_myfloor.append(m)
    idx = 0
    for n, m in model.named_modules():
        if isinstance(m, module):
            m.relu = modules_myfloor[idx]
            m.threshold = modules_myfloor[idx].up
            print(m.threshold)
            idx += 1
    return modules_myfloor

def replace_myfloor_by_StraightThrough(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules") and module.__class__.__name__ != 'SpikeModule' and module.__class__.__name__ != 'SpikeResModule':
            model._modules[name] = replace_myfloor_by_StraightThrough(module)
        if module.__class__.__name__ == 'MyFloor':
            model._modules[name] = StraightThrough()
    return model