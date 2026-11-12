import argparse
from experiment_config import checkpoint, load_model_state
import os
import math
import timm
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

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
            model_path = checkpoint("test_model/alexnet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/alexnet/"


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = checkpoint("test_model/vit_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/vit/"

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
            model_path = checkpoint("test_model/alexnet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/vit/"

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
            model_path = checkpoint("test_model/alexnet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/resnet18/"

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/vit/"

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
            model_path = checkpoint("test_model/alexnet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/resnet18/"

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/vit/"

    return model, samples_dir


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


def acc_test(test_loader, model, target_label):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, _ in test_loader:
            predicted = model(images).argmax(dim=1)
            correct += (predicted == target_label).sum().item()
    total = len(test_loader.dataset)
    print(f"Fingerprint label agreement: {correct}/{total} ({100 * correct / total:.2f}%)")


def main():
    datasets = {"fashion-mnist": 1, "cifar10": 2, "tiny-imagenet": 3, "gtsrb": 4}
    architectures = {"alexnet": 1, "vgg16": 2, "resnet18": 3, "mobilenet": 4, "vit": 5}
    parser = argparse.ArgumentParser(description="Evaluate semi-fragile fingerprints on original and modified models.")
    parser.add_argument("--dataset", choices=datasets, default="cifar10")
    parser.add_argument("--architecture", choices=architectures, default="resnet18")
    parser.add_argument("--target-label", type=int, default=3)
    parser.add_argument("--samples-root", help="ImageFolder root of the fingerprint images")
    parser.add_argument("--limit", type=int, help="Evaluate the first N images")
    parser.add_argument("--finetune-epochs", type=int, nargs="+", default=[])
    parser.add_argument("--pruning-rates", type=float, nargs="+", default=[])
    parser.add_argument("--badnet-runs", type=int, nargs="+", default=[])
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    dataset_idx = datasets[args.dataset]
    model_idx = architectures[args.architecture]
    target_model, samples_dir = model_selection(model_idx, dataset_idx)
    resize = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if dataset_idx == 1:
        resize.insert(0, transforms.Grayscale(num_output_channels=1))
    dataset = ImageFolder(root=args.samples_root or samples_dir, transform=transforms.Compose(resize))
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Original model")
    acc_test(loader, target_model, args.target_label)
    quantized_model = torch.quantization.quantize_dynamic(
        target_model, {torch.nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    print("Dynamically quantized model")
    acc_test(loader, quantized_model, args.target_label)

    for epochs in args.finetune_epochs:
        print(f"Fine-tuned model ({epochs} epochs)")
        acc_test(loader, benign_model_selection(model_idx, dataset_idx, epochs, 1), args.target_label)
    for rate in args.pruning_rates:
        print(f"Pruned model (rate {rate})")
        acc_test(loader, benign_model_selection(model_idx, dataset_idx, rate, 2), args.target_label)
    for run in args.badnet_runs:
        print(f"BadNet model (run {run})")
        acc_test(loader, poisoned_model_selection(model_idx, dataset_idx, run, 1), args.target_label)


if __name__ == "__main__":
    main()
