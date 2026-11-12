import os
import math
import timm
import copy
import torch
import random
import warnings
import itertools
import numpy as np
import torchvision
import torch.nn as nn
import os.path as osp
# from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from adversarial_sample_generation import *
import torchvision.transforms as transforms
from Tinyimagenet_dataset import TinyImageNet
from skimage.measure import compare_ssim as ssim

# from torch.utils.tensorboard import SummaryWriter

# ================================<<环境设备设置>>================================ #
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ================================<<环境设备设置>>================================ #
def model_selection(model_idx, dataset_idx):
    if dataset_idx == 1:
        # print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
        # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # train_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
        #                                                   train=True, transform=transform)
        # test_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
        #                                                  train=False, transform=transform)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_fash/alexnet/"
            logit_path = './logit_data/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'alexnet_fash'


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_fash/vgg16/"
            logit_path = './logit_data/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vgg16_fash'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_fash/resnet18/"
            logit_path = './logit_data/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'resnet18_fash'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_fash/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_fash/vit/"
            logit_path = './logit_data/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vit_fash'

    elif dataset_idx == 2:
        # print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])
        # train_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=True,
        #                                              transform=transform)
        # test_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=False,
        #                                             transform=transform)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_cifar/alexnet/"
            # samples_dir = "./semi-fragile_samples_cifar/alexnet/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'alexnet_cifar'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_cifar/vgg16/"
            # samples_dir = "./semi-fragile_samples_cifar/vgg16/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vgg16_cifar'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_cifar/resnet18/"
            # samples_dir = "./semi-fragile_samples_cifar/resnet18/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'resnet18_cifar'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_cifar/mobilenet/"

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            # samples_dir = "./selected_samples_cifar/vit/"
            # samples_dir = "./semi-fragile_samples_cifar/vit/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vit_cifar'

    elif dataset_idx == 3:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # data_dir = r'D:\HL\project\dataset\tiny-imagenet-200'
        # train_dataset = TinyImageNet(data_dir, train=True, transform=transform)
        # test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_tiny/alexnet/"
            # samples_dir = "./semi-fragile_samples_tiny/alexnet/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'alexnet_tiny'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_tiny/vgg16/"
            # samples_dir = "./semi-fragile_samples_tiny/vgg16/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vgg16_tiny'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_tiny/resnet18/"
            # samples_dir = "./semi-fragile_samples_tiny/resnet18/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'resnet18_tiny'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_tiny/mobilenet/"

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_tiny/vit/"
            # samples_dir = "./semi-fragile_samples_tiny/vit/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vit_tiny'

    elif dataset_idx == 4:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 43
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # data_dir = r'D:\HL\project\dataset\tiny-imagenet-200'
        # train_dataset = TinyImageNet(data_dir, train=True, transform=transform)
        # test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_gtsrb/alexnet/"
            # samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'alexnet_gtsrb'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_gtsrb/vgg16/"
            # samples_dir = "./semi-fragile_samples_gtsrb/vgg16/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vgg16_gtsrb'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_gtsrb/resnet18/"
            # samples_dir = "./semi-fragile_samples_gtsrb/resnet18/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'resnet18_gtsrb'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_gtsrb/mobilenet/"

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./selected_samples_gtsrb/vit/"
            # samples_dir = "./semi-fragile_samples_gtsrb/vit/"
            logit_path = './logit_data_semi/'
            os.makedirs(logit_path, exist_ok=True)
            logit_path = logit_path + 'vit_gtsrb'

    return model, samples_dir, logit_path, num_classes


def benign_model_selection(model_idx, dataset_idx, pruning_idx, benign_idx):
    if dataset_idx == 1:
        # print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/alexnet_fash_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/alexnet_fash_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/vgg16_fash_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/vgg16_fash_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_fash_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_fash_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_fash_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_fash_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/VIT_fash_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/VIT_fash_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif dataset_idx == 2:
        # print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/alexnet_cifar_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/alexnet_cifar_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/vgg16_cifar_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/vgg16_cifar_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_cifar_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_cifar_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/alexnet_cifar_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/alexnet_cifar_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/VIT_cifar_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/VIT_cifar_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif dataset_idx == 3:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/alexnet_tiny_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/alexnet_tiny_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/vgg16_tiny_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/vgg16_tiny_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_tiny_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_tiny_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_tiny_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_tiny_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/VIT_tiny_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/VIT_tiny_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif dataset_idx == 4:
        # print('# =====================<<Dataset is GTSRB>>===================== #')
        num_classes = 43
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/alexnet_gtsrb_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/alexnet_gtsrb_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/vgg16_gtsrb_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/vgg16_gtsrb_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_gtsrb_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_gtsrb_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/resnet18_gtsrb_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/resnet18_gtsrb_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/VIT_gtsrb_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/VIT_gtsrb_Conv1_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    return model


def poisoned_model_selection(model_idx, dataset_idx, pruning_idx, poisoned_idx):
    if dataset_idx == 1:
        # print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_fash_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_fash_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_fash_blended_' \
                             + str(pruning_idx)

            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/vgg16_fash_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/vgg16_fash_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/vgg16_fash_blended_' \
                             + str(pruning_idx)

            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/resnet18_fash_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/resnet18_fash_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/resnet18_fash_blended_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_fash_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_fash_WaNet_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/VIT_fash_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/VIT_fash_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/VIT_fash_blended_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif dataset_idx == 2:
        # print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/cifar/alexnet_cifar_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/cifar/alexnet_cifar_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/cifar/alexnet_cifar_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/cifar/alexnet_cifar_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/cifar/vgg16_cifar_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/cifar/vgg16_cifar_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/cifar/vgg16_cifar_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/cifar/vgg16_cifar_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/cifar/resnet18_cifar_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/cifar/resnet18_cifar_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/cifar/resnet18_cifar_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/cifar/resnet18_cifar_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/mobilenet/mobilenet_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/mobilenet/mobilenet_WaNet_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/cifar/VIT_cifar_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/cifar/VIT_cifar_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/cifar/VIT_cifar_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/cifar/vit_cifar_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif dataset_idx == 3:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_tiny_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_tiny_WaNet_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/VIT_tiny_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/VIT_tiny_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/VIT_tiny_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/vit_tiny_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif dataset_idx == 4:
        # print('# =====================<<Dataset is Tiny GTSRB>>===================== #')
        num_classes = 43
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/gtsrb/vgg16_gtsrb_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/gtsrb/vgg16_gtsrb_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/gtsrb/vgg16_gtsrb_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/gtsrb/vgg16_gtsrb_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_gtsrb_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_gtsrb_WaNet_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/VIT_gtsrb_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/VIT_gtsrb_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/VIT_gtsrb_blended_' \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/vit_gtsrb_issba_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    return model


def acc_test(test_loader, model, num_classes):
    model.eval()
    with torch.no_grad():
        output_matrix = np.zeros((100, num_classes))
        t = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)
            outputs = model(images)
            output_vector = F.normalize(outputs, p=2, dim=1)
            output_tem = output_vector.cpu().numpy()
            output_matrix[t, :] = output_tem
            t += 1
    return output_matrix


def main():
    batch_size = 1
    dataset_idx = 2  # 1: FashionMNST; 2: CIFAR-10; 3: Tiny-ImageNet; 4: GTSRB
    model_idx_list = [1, 2, 3]  # 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT
    fintuning_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    pruning_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    backdoor_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for model_idx in model_idx_list:
        print(
            f'# =====================================<<Model_idx:{model_idx}>>===================================== #')
        target_model, samples_dir, logit_path, num_classes = model_selection(model_idx, dataset_idx)
        if dataset_idx == 1:
            data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                 transforms.Resize((224, 224)), transforms.ToTensor()])
        else:
            data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset = ImageFolder(root=samples_dir, transform=data_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        output1 = []
        output11 = acc_test(dataloader, target_model, num_classes)
        output1.append(output11)
        print('# =======================<<Fintuned Model>>======================= #')
        output2_1 = []
        output2_2 = []
        for fintuning_idx in fintuning_list:
            print('The model index: {}, | the fintuning index: {}'.format(model_idx, fintuning_idx))
            benign_idx = 1
            benign_model = benign_model_selection(model_idx, dataset_idx, fintuning_idx, benign_idx)
            output21 = acc_test(dataloader, benign_model, num_classes)
            output22 = output21 - output11
            output2_1.append(output21)
            output2_2.append(output22)

        print('# =======================<<Pruning Model>>======================= #')
        output3_1 = []
        output3_2 = []
        for pruning_idx in pruning_list:
            print('The model index: {}, | the pruning index: {}'.format(model_idx, pruning_idx))
            benign_idx = 2
            benign_model = benign_model_selection(model_idx, dataset_idx, pruning_idx, benign_idx)
            output31 = acc_test(dataloader, benign_model, num_classes)
            output32 = output31 - output11
            output3_1.append(output31)
            output3_2.append(output32)

        print('# =======================<<Poisoned Model>>======================= #')
        output4_1 = []
        output4_2 = []
        for backdoor_idx in backdoor_list:
            print('The model index: {}, | the BadNet attack index: {}'.format(model_idx, backdoor_idx))
            poisoned_idx = 1
            poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
            output41 = acc_test(dataloader, poisoned_model, num_classes)
            output42 = output41 - output11
            output4_1.append(output41)
            output4_2.append(output42)

        output5_1 = []
        output5_2 = []
        for backdoor_idx in backdoor_list:
            print('The model index: {}, | the WaNet attack index: {}'.format(model_idx, backdoor_idx))
            poisoned_idx = 2
            poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
            output51 = acc_test(dataloader, poisoned_model, num_classes)
            output52 = output51 - output11
            output5_1.append(output51)
            output5_2.append(output52)

        output6_1 = []
        output6_2 = []
        for backdoor_idx in backdoor_list:
            print('The model index: {}, | the Blended attack index: {}'.format(model_idx, backdoor_idx))
            poisoned_idx = 3
            poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
            output61 = acc_test(dataloader, poisoned_model, num_classes)
            output62 = output61 - output11
            output6_1.append(output61)
            output6_2.append(output62)

        output7_1 = []
        output7_2 = []
        for backdoor_idx in backdoor_list:
            print('The model index: {}, | the ISSBA attack index: {}'.format(model_idx, backdoor_idx))
            poisoned_idx = 4
            poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
            output71 = acc_test(dataloader, poisoned_model, num_classes)
            output72 = output71 - output11
            output7_1.append(output71)
            output7_2.append(output72)

        output8_1 = []
        output8_2 = []
        for backdoor_idx in backdoor_list:
            print('The model index: {}, | the IFSR attack index: {}'.format(model_idx, backdoor_idx))
            poisoned_idx = 4
            poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
            output81 = acc_test(dataloader, poisoned_model, num_classes)
            output82 = output81 - output11
            output8_1.append(output81)
            output8_2.append(output82)

        output9_1 = []
        output9_2 = []
        for backdoor_idx in backdoor_list:
            print('The model index: {}, | the LC attack index: {}'.format(model_idx, backdoor_idx))
            poisoned_idx = 4
            poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
            output91 = acc_test(dataloader, poisoned_model, num_classes)
            output92 = output91 - output11
            output9_1.append(output91)
            output9_2.append(output92)

        # output_list1 = output1 + output2_1 + output3_1 + output4_1 + output5_1 + output6_1  # 所有模型的输出向量
        output_list1 = output1 + output2_1 + output4_1 + output5_1 + output6_1 + output7_1 + output8_1 + output9_1  # 除剪枝模型外所有模型的输出向量
        output_list2 = output2_2 + output4_2 + output5_1 + output6_2
        output_numpy1 = np.array(list(itertools.chain(*output_list1)))

        label_ori1 = np.ones((100, 1)) * 0
        label_ori2 = np.ones((1000, 1)) * 1
        label_poi1 = np.ones((1000, 1)) * 2
        label_poi2 = np.ones((1000, 1)) * 3
        label_poi3 = np.ones((1000, 1)) * 4
        label_poi4 = np.ones((1000, 1)) * 5

        label = np.vstack((label_ori1, label_ori2, label_poi1, label_poi2, label_poi3, label_poi4))
        logit_vector = np.hstack((output_numpy1, label))
        np.save('{}.npy'.format(logit_path), logit_vector)

        output_numpy2 = np.array(list(itertools.chain(*output_list2)))  # 微调模型与中毒模型输出向量与原始模型输出向量的差值
        E_finetuning = np.mean(output_numpy2[0:1000, :])
        V_finetuning = np.std(output_numpy2[0:1000, :])
        print('The mean of Fine-tuning model:{:.4f}, The std of Fine-tuning model:{:.4f}'.format(E_finetuning,
                                                                                                 V_finetuning))
        E_poisoned1 = np.mean(output_numpy2[1000:2000, :])
        V_poisoned1 = np.std(output_numpy2[1000:2000, :])
        print('The mean of Poisoned1 model:{:.4f}, The std of Poisoned1 model:{:.4f}'.format(E_poisoned1, V_poisoned1))
        E_poisoned2 = np.mean(output_numpy2[2000:3000, :])
        V_poisoned2 = np.std(output_numpy2[2000:3000, :])
        print('The mean of Poisoned2 model:{:.4f}, The std of Poisoned2 model:{:.4f}'.format(E_poisoned2, V_poisoned2))
        E_poisoned3 = np.mean(output_numpy2[3000:4000, :])
        V_poisoned3 = np.std(output_numpy2[3000:4000:, :])
        print('The mean of Poisoned3 model:{:.4f}, The std of Poisoned3 model:{:.4f}'.format(E_poisoned3, V_poisoned3))
        E_poisoned_total = np.mean(output_numpy2[1000:, :])
        V_poisoned_total = np.std(output_numpy2[1000:, :])
        print('The mean of Poisoned model:{:.4f}, The std of Poisoned model:{:.4f}'.format(E_poisoned_total,
                                                                                           V_poisoned_total))
        E_poisoned_non_ori = np.mean(output_numpy1[100:, :])
        V_poisoned_non_ori = np.std(output_numpy1[100:, :])
        print('The mean of Non-original model:{:.4f}, The std of Non-original model:{:.4f}'.format(E_poisoned_non_ori,
                                                                                                   V_poisoned_non_ori))


if __name__ == '__main__':
    main()
