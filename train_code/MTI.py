from experiment_config import checkpoint, load_classifier, load_model_state
import os
import math
import timm
import copy
import torch
import random
import warnings
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
import os.path as osp
# from tqdm import tqdm
from torch import optim
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from adversarial_sample_generation import *
import torchvision.transforms as transforms
from Tinyimagenet_dataset import TinyImageNet

# from skimage.measure import compare_ssim as ssim

# from torch.utils.tensorboard import SummaryWriter

# ================================<<环境设备设置>>================================ #
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ================================<<环境设备设置>>================================ #
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        hidden_vec = self.layers(x)
        output = self.fc(hidden_vec)
        return output


def model_selection(model_idx, dataset_idx):
    classifier_path = './finetuning_classifier/'
    if dataset_idx == 1:
        print('# =====================<<Dataset is FashionMNIST>>===================== #')
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
            model_path = checkpoint("test_model/alexnet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/alexnet/"


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = checkpoint("test_model/vit_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/vit/"

    elif dataset_idx == 2:
        print('# =====================<<Dataset is Cifar-10>>===================== #')
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
            model_path = checkpoint("test_model/alexnet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/alexnet/"
            classifier_path = classifier_path + 'alexnet_cifar_fine-tuned_v1'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vgg16/"
            classifier_path = classifier_path + 'vgg16_cifar_fine-tuned_v1'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/resnet18/"
            classifier_path = classifier_path + 'resnet18_cifar_fine-tuned_v1'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vit/"
            classifier_path = classifier_path + 'vit_cifar_fine-tuned_v1'

    elif dataset_idx == 3:
        print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
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
            model_path = checkpoint("test_model/alexnet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/alexnet/"
            classifier_path = classifier_path + 'alexnet_tiny_fine-tuned_v1'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vgg16/"
            classifier_path = classifier_path + 'vgg16_tiny_fine-tuned_v1'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/resnet18/"
            classifier_path = classifier_path + 'resnet18_tiny_fine-tuned_v1'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vit/"
            classifier_path = classifier_path + 'vit_tiny_fine-tuned_v1'

    elif dataset_idx == 4:
        print('# =====================<<Dataset is GTSRB>>===================== #')
        num_classes = 43
        # target_vector = torch.zeros(num_classes)
        # target_vector[target_label] = 1.0
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        # ])
        # data_dir = r'D:\HL\project\dataset\GTSRB1\train'
        # dataset = ImageFolder(root=data_dir, transform=transform)
        # torch.manual_seed(126)
        # train_sampler = torch.randperm(len(dataset))
        # train_indices = train_sampler[int(len(dataset) * 0.1):]
        # test_indices = train_sampler[:int(len(dataset) * 0.1)]
        # train_dataset = torch.utils.data.Subset(dataset, train_indices)
        # test_dataset = torch.utils.data.Subset(dataset, test_indices)
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/alexnet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"
            classifier_path = classifier_path + 'alexnet_gtsrb_fine-tuned_v1'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/vgg16/"
            classifier_path = classifier_path + 'vgg16_gtsrb_fine-tuned_v1'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/resnet18/"
            classifier_path = classifier_path + 'resnet18_gtsrb_fine-tuned_v1'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/vit/"
            classifier_path = classifier_path + 'vit_gtsrb_fine-tuned_v1'

    return model, samples_dir, sel_dir, classifier_path


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
                model_path = checkpoint("fintuned_model/alexnet_fash_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/alexnet_fash_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/vgg16_fash_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/vgg16_fash_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_fash_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_fash_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_fash_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_fash_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/VIT_fash_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/VIT_fash_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 2:
        # print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/alexnet_cifar_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/alexnet_cifar_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/vgg16_cifar_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/vgg16_cifar_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_cifar_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_cifar_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/alexnet_cifar_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/alexnet_cifar_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/VIT_cifar_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/VIT_cifar_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 3:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/alexnet_tiny_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/alexnet_tiny_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/vgg16_tiny_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/vgg16_tiny_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_tiny_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_tiny_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_tiny_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_tiny_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/VIT_tiny_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/VIT_tiny_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 4:
        # print('# =====================<<Dataset is GTSRB>>===================== #')
        num_classes = 43
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/alexnet_gtsrb_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/alexnet_gtsrb_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/vgg16_gtsrb_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/vgg16_gtsrb_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_gtsrb_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_gtsrb_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/resnet18_gtsrb_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/resnet18_gtsrb_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = checkpoint("fintuned_model/VIT_gtsrb_") \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = checkpoint("pruning_model/VIT_gtsrb_Conv1_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

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
                model_path = checkpoint("backdoor_model/alexnet/alexnet_fash_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_fash_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_fash_blended_") \
                             + str(pruning_idx)

            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vgg16/vgg16_fash_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vgg16/vgg16_fash_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vgg16/vgg16_fash_blended_") \
                             + str(pruning_idx)

            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/resnet18/resnet18_fash_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/resnet18/resnet18_fash_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/resnet18/resnet18_fash_blended_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_fash_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_fash_WaNet_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vit/VIT_fash_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vit/VIT_fash_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vit/VIT_fash_blended_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 2:
        # print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_LC_") \
                             + str(pruning_idx)

            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_LC_") \
                             + str(pruning_idx)

            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/mobilenet/mobilenet_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/mobilenet/mobilenet_WaNet_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vit/cifar/VIT_cifar_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vit/cifar/VIT_cifar_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vit/cifar/VIT_cifar_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/vit/cifar/vit_cifar_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/vit/cifar/vit_cifar_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/vit/cifar/vit_cifar_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 3:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/resnet18/tiny/resnet18_tiny_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/resnet18/tiny/resnet18_tiny_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/resnet18/tiny/resnet18_tiny_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/resnet18/tiny/resnet18_tiny_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/resnet18/tiny/resnet18_tiny_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/resnet18/tiny/resnet18_tiny_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_tiny_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_tiny_WaNet_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vit/tiny/VIT_tiny_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vit/tiny/VIT_tiny_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vit/tiny/VIT_tiny_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/vit/tiny/vit_tiny_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/vit/tiny/vit_tiny_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/vit/tiny/vit_tiny_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 4:
        # print('# =====================<<Dataset is Tiny GTSRB>>===================== #')
        num_classes = 43
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_gtsrb_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/alexnet/alexnet_gtsrb_WaNet_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if poisoned_idx == 1:
                model_path = checkpoint("backdoor_model/vit/gtsrb/VIT_gtsrb_badnet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = checkpoint("backdoor_model/vit/gtsrb/VIT_gtsrb_WaNet_") \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = checkpoint("backdoor_model/vit/gtsrb/VIT_gtsrb_blended_") \
                             + str(pruning_idx)
            elif poisoned_idx == 4:
                model_path = checkpoint("backdoor_model/vit/gtsrb/vit_gtsrb_issba_") \
                             + str(pruning_idx)
            elif poisoned_idx == 5:
                model_path = checkpoint("backdoor_model/vit/gtsrb/vit_gtsrb_ifsr_") \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = checkpoint("backdoor_model/vit/gtsrb/vit_gtsrb_LC_") \
                             + str(pruning_idx)
            model.load_state_dict(load_model_state(model_path))

    return model


def layer_train(train_loader, mapping_layer, criterion, opt, epoch):
    for i, data in enumerate(train_loader):
        opt.zero_grad()
        logit_vec, labels = data
        labels = labels.to(device)
        logit_vec = logit_vec.to(device)
        mapping_layer = mapping_layer.to(device)
        output = mapping_layer(logit_vec)
        loss = criterion(output, labels.long())
        loss.backward()
        opt.step()
        print(' Epoch/batch_idx: {}/{}, | the loss: {:.5f}'.format(epoch, i, loss.item()))
    return mapping_layer


def acc_test(test_loader, model, classifier_model, target_lable, cor_num):
    model.eval()
    with torch.no_grad():
        correct = 0
        tensor_list = torch.zeros(len(test_loader))
        for images, labels in test_loader:
            outputs = classifier_model(F.normalize(model(images), p=2, dim=1))
            _, predicted = torch.max(outputs.data, 1)
            tensor_list[correct] = predicted[0]
            correct = correct + 1
            # if correct >= 6:
            #     break

        unique_values, counts = torch.unique(tensor_list, return_counts=True)
        # print('the predicted label types: {}, | the corresponding counts: {}'.format(unique_values, counts))
        max_count_index = torch.argmax(counts)
        # print('The target label is :{}, | the predicted label is :{}'.format(target_lable,
        #                                                                      unique_values[max_count_index]))
    if target_lable == int(unique_values[max_count_index]):
        # print('# =====================================<<MTI is success!!!>>===================================== #')
        cor_num = 1
    else:
        print('# =========================<<MTI is error!!!>>========================= #')
    return cor_num


def main():
    # =====================<<使用投票机制来识别对于每种修改类型>>===================== #
    batch_size = 1
    dataset_idx_list = [2, 3, 4]  # 1: FashionMNST; 2: CIFAR-10; 3: Tiny-ImageNet; 4: GTSRB
    model_idx_list = [1, 2, 3, 5]  # 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT
    fintuning_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # pruning_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pruning_list = [0.1, 0.2]
    backdoor_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for dataset_idx in dataset_idx_list:
        for model_idx in model_idx_list:
            target_model, samples_dir, sel_dir, classifier_path = model_selection(model_idx, dataset_idx)
            classifier_model = load_classifier('{}.pth'.format(classifier_path), ProjectionHead)
            classifier_model.cpu()
            quantized_model = torch.quantization.quantize_dynamic(target_model, {torch.nn.Linear, nn.Conv2d},
                                                                  dtype=torch.qint8)
            quantized_model = quantized_model
            if dataset_idx == 1:
                data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                     transforms.Resize((224, 224)), transforms.ToTensor()])
            else:
                data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            dataset = ImageFolder(root=samples_dir, transform=data_transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print('The model is original model')
            count1 = 0
            target_lable = 0
            output1 = acc_test(dataloader, target_model, classifier_model, target_lable, count1)
            count1 += output1
            print('The model is quantized model')
            count2 = 0
            target_lable = 1
            output2 = acc_test(dataloader, quantized_model, classifier_model, target_lable, count2)
            count2 += output2
            print('\n')
            print('# =======================<<Fintuned Model>>======================= #')
            count3 = 0
            for fintuning_idx in fintuning_list:
                print('The model index: {}, | the fintuning index: {}'.format(model_idx, fintuning_idx))
                benign_idx = 1
                target_lable = 2
                benign_model = benign_model_selection(model_idx, dataset_idx, fintuning_idx, benign_idx)
                output3 = acc_test(dataloader, benign_model, classifier_model, target_lable, count3)
                count3 += output3

            print('\n')
            print('# =======================<<Pruning Model>>======================= #')
            count4 = 0
            for pruning_idx in pruning_list:
                print('The model index: {}, | the pruning index: {}'.format(model_idx, pruning_idx))
                benign_idx = 2
                target_lable = 3
                benign_model = benign_model_selection(model_idx, dataset_idx, pruning_idx, benign_idx)
                output4 = acc_test(dataloader, benign_model, classifier_model, target_lable, count4)
                count4 += output4

            print('\n')
            print('# =======================<<Poisoned Model>>======================= #')
            count5 = 0
            count6 = 0
            count7 = 0
            count8 = 0
            count9 = 0
            count10 = 0
            for backdoor_idx in backdoor_list:
                print('The model index: {}, | the BadNet attack index: {}'.format(model_idx, backdoor_idx))
                poisoned_idx = 1
                target_lable = 4
                poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
                output5 = acc_test(dataloader, poisoned_model, classifier_model, target_lable, count5)
                count5 += output5
                print('The model index: {}, | the WaNet attack index: {}'.format(model_idx, backdoor_idx))
                poisoned_idx = 2
                target_lable = 5
                poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
                output6 = acc_test(dataloader, poisoned_model, classifier_model, target_lable, count6)
                count6 += output6
                print('The model index: {}, | the Blended attack index: {}'.format(model_idx, backdoor_idx))
                poisoned_idx = 3
                target_lable = 6
                poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
                output7 = acc_test(dataloader, poisoned_model, classifier_model, target_lable, count7)
                count7 += output7
                print('The model index: {}, | the ISSBA attack index: {}'.format(model_idx, backdoor_idx))
                poisoned_idx = 4
                target_lable = 7
                poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
                output8 = acc_test(dataloader, poisoned_model, classifier_model, target_lable, count8)
                count8 += output8
                print('The model index: {}, | the IFSR attack index: {}'.format(model_idx, backdoor_idx))
                poisoned_idx = 5
                target_lable = 8
                poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
                output9 = acc_test(dataloader, poisoned_model, classifier_model, target_lable, count9)
                count9 += output9
                print('The model index: {}, | the LC attack index: {}'.format(model_idx, backdoor_idx))
                poisoned_idx = 6
                target_lable = 9
                poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
                output10 = acc_test(dataloader, poisoned_model, classifier_model, target_lable, count10)
                count10 += output10
                print('\n')

            print('# ======================The dataset index is: {}, | The model index is: {} ====================== #'.format(
                dataset_idx, model_idx))
            print('The accuracy of MTI: Original: {}, | Quan: {}, Fine-tuning: {}, | Pruning: {}'.format(count1, count2,
                                                                                                         count3,
                                                                                                         count4))
            print('The accuracy of MTI: BadNet: {}, | WaNet: {}, Blended: {}, | ISSBA: {}, IBA: {}, | LC: {}'.format(
                count5,
                count6,
                count7,
                count8,
                count9,
                count10))
            print(
                '# ======================The dataset index is: {}, | The model index is: {} ====================== #'.format(
                    dataset_idx, model_idx))
            print('\n')


if __name__ == '__main__':
    main()
