import torch
import argparse
import random
import numpy as np
import os
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data.autoaugment import CIFAR10Policy, Cutout
from CIFAR.models.vgg import VGG
from CIFAR.models.resnet import resnet20
from models.utils import *
from models.spiking_layer import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.rcParams['savefig.dpi'] = 300 #图片像素


def seed_all(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data(dpath: str, batch_size=128, cutout=False, workers=0, use_cifar10=False, auto_aug=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=dpath, train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=dpath, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16', 'res20'])
    # parser.add_argument('--dpath', default='D:\Dataset\cifar10', required=True, type=str, help='dataset directory')
    parser.add_argument('--dpath', default='Dataset', type=str, help='dataset directory')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=2, type=int, help='number of training epochs')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (L2 regularization)')
    parser.add_argument('--usebn',  default=True, type=bool, help='use batch normalization in ann') #使用BN

    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs

    use_cifar10 = args.dataset == 'CIFAR10' #True = CIFAR-10 False = CIFAR-100
    train_loader, test_loader = build_data(dpath=args.dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=True)
    # train_loader, test_loader = build_data(dpath=args.dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=False)
    best_acc = 0
    best_epoch = 0
    use_bn = args.usebn
    args.wd = 5e-4 if use_bn else 1e-4
    bn_name = 'wBN' if use_bn else 'woBN'
    # model_save_name = '../resnet20_cifar100-Phase.pth'
    # model_save_name = '../saved_models/ReLU1-ImageNet-VGG-16.pth'
    model_save_name = '../saved_model/ReLU-CIFAR-10-L=8.pth'

    if args.arch == 'VGG16':
        ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
    elif args.arch == 'res20':
        ann = resnet20(use_bn=use_bn,  num_classes=10 if use_cifar10 else 100)
        # ann = resnet20(use_bn=use_bn, num_class=10 if use_cifar10 else 100)
    else:
        raise NotImplementedError
    print(use_bn)
    criterion = nn.CrossEntropyLoss().to(device)
    ann.load_state_dict(torch.load('../saved_models/ReLU-CIFAR-10-95.89.pth'))
    ann = replace_activation_by_floor(ann, t=8) #进行量化-ANN
    # ann.load_state_dict(torch.load('../saved_models/ResNet20-ReLU-CIFAR-10-PhaseII.pth')) #测试量化ANN的精度
    print(ann)

    # build optimizer #训练ANN使用的优化器
    # optimizer = torch.optim.SGD(ann.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) if use_bn else \
    #             torch.optim.SGD(ann.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    #训练量化ANN使用的优化器
    optimizer = torch.optim.SGD(ann.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    ann.to(device)

    plt.switch_backend('Agg')
    Loss = []
    Loss1 = []

    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)
            outputs = ann(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 40 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, running_loss))
                # running_loss = 0
                print('Time elapsed:', time.time() - start_time)

        Loss.append(running_loss / 1000.0)
        Loss1.append(running_loss / 1000 + 0.5)
        running_loss = 0

        scheduler.step()
        correct = 0
        total = 0

        # start testing
        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = ann(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets.cpu()).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

        print('Iters:', epoch)
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            # torch.save(ann.state_dict(), model_save_name)
        print('best_acc is: ', best_acc, ' find in epoch: ', best_epoch)
        print('\n\n')

        plt.figure()
        plt.plot(Loss, '#57585A', label='L=8')
        plt.plot(Loss1, 'gray', label='L=8')
        plt.ylabel('Loss(×1000)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig("../saved_models/1_recon_loss.jpg", dpi=300)


