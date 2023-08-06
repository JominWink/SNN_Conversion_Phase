import torch
import torch.nn as nn
import copy
import time
import os
import random
import argparse
import numpy as np
from main_train import build_data
from CIFAR.models.vgg import VGG
from CIFAR.models.resnet import resnet20, res_specials
from CIFAR.models.fold_bn import search_fold_and_remove_bn
from CIFAR.models.spiking_layer import SpikeModel, get_maximum_activation, SpikeModule
from models.utils import replace_activation_by_floor, set_threshold_by_Myfloor, replace_myfloor_by_StraightThrough
from models.calibration import bias_corr_model, weights_cali_model

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def validate_model(test_loader, ann): #测试SNN网络
    correct = 0
    total = 0
    ann.eval()
    device = next(ann.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = ann(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='res20', type=str, help='network architecture', choices=['VGG16', 'res20'])
    parser.add_argument('--dpath',  default='Dataset', type=str, help='dataset directory')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--calib', default='light', type=str, help='calibration methods', choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=8, type=int, help='snn simulation length')
    parser.add_argument('--l', default=4, type=int, help='L')
    # parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--usebn', default=True, type=bool, help='use batch normalization in ann')  # 使用BN
    args = parser.parse_args()
    results_list = []
    use_bn = args.usebn

    # we run the experiments for 5 times, with different random seeds
    for_epoch = 1
    for i in range(for_epoch):

        seed_all(seed=args.seed + i)
        sim_length = 4

        use_cifar10 = args.dataset == 'CIFAR10'

        train_loader, test_loader = build_data(dpath=args.dpath, batch_size=args.batch_size, cutout=True, use_cifar10=use_cifar10, auto_aug=True)

        if args.arch == 'VGG16':
            ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
        elif args.arch == 'res20':
            ann = resnet20(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
            # ann = resnet20(use_bn=use_bn, num_class=10 if use_cifar10 else 100)
        else:
            raise NotImplementedError

        args.wd = 5e-4 if use_bn else 1e-4
        bn_name = 'wBN' if use_bn else 'woBN'
        print(args.l)
        ann = replace_activation_by_floor(ann, t=args.l)
        # load_path = '../resnet20_cifar100-Phase.pth'
        # load_path = '../saved_models/ReLU-CIFAR-10-PhaseI-95.88.pth'
        # load_path = '../saved_models/ResNet20-ReLU-CIFAR-10-PhaseI.pth'
        # load_path = '../saved_models/ReLU-CIFAR-100-PhaseI-77.22.pth'
        load_path = '../saved_models/res20-CIFAR-100-L=4.pth'
        state_dict = torch.load(load_path, map_location=torch.device('cpu')) #加载保存好的模型
        ann.load_state_dict(state_dict, strict=True)
        print(ann)
        search_fold_and_remove_bn(ann)
        ann.cuda()
        print(ann)
        snn = SpikeModel(model=ann, sim_length=sim_length, specials=res_specials)
        snn.cuda()
        print(snn)
        module_myfloor = set_threshold_by_Myfloor(model=snn, module=SpikeModule)
        replace_myfloor_by_StraightThrough(model=snn)
        print(snn)

        if args.calib == 'light':
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
            weights_cali_model(model=snn, train_loader=train_loader, num_cali_samples=1024, learning_rate=1e-5)
            # train_PhaseII_snn(model=snn, module=SpikeModule, train_dataloader=train_loader, test_dataloader=test_loader)
        snn.set_spike_state(use_spike=True)
        results_list.append(validate_model(test_loader, snn))

    a = np.array([results_list])
    print(a.mean(), a.std())
