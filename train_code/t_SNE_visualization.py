import os
import math
import timm
import torch
import warnings
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.quantization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
# from adversarial_sample_generation import *
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
print(device)


def model_selection(model_idx, dataset_idx):
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/alexnet/"


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vit/"

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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/resnet18/"

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vit/"

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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"

    return model, samples_dir, sel_dir


def benign_model_selection(model_idx, dataset_idx, pruning_idx, benign_idx):
    if dataset_idx == 1:
        # print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
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
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/alexnet_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/alexnet_' \
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
        num_classes = 43
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
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
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\fintuned_model/mobilenet_gtsrb_5'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            if benign_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/fintuned_model/vit_gtsrb_' \
                             + str(pruning_idx)
            elif benign_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme/pruning_model/vit_gtsrb_Conv1_' \
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
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/alexnet_fash_issba_' \
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
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/vgg16_fash_issba_' \
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
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/resnet18_fash_issba_' \
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
            elif poisoned_idx == 4:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/VIT_fash_issba_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/cifar/alexnet_cifar_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/cifar/alexnet_cifar_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/cifar/vgg16_cifar_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/cifar/vgg16_cifar_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/cifar/resnet18_cifar_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/cifar/resnet18_cifar_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/cifar/vit_cifar_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/cifar/vit_cifar_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_LC_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_WaNet_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/vit_tiny_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/vit_tiny_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/gtsrb/vgg16_gtsrb_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/gtsrb/vgg16_gtsrb_LC_' \
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_LC_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            if poisoned_idx == 1:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_badnet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 2:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/gtsrb/alexnet_gtsrb_WaNet_' \
                             + str(pruning_idx)
            elif poisoned_idx == 3:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/gtsrb/resnet18_gtsrb_blended_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

        else:
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/vit_gtsrb_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/vit_gtsrb_LC_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    return model


def main():
    batch_size = 1
    # =======================================<<Parameter Modifying>>======================================= #
    dataset_idx = 2  # 1: FashionMNST; 2: Cifar10; 3: Tiny-ImageNet
    vector_idx = 1  # 1表示logit输出，2表示softmax输出
    samples_idx = 2  # 1:原始挑选样本; 2:生成的半脆弱样本
    model_idx_list = [1, 2, 3]  # 目标模型（需要保护的模型）加载 1:alexnet; 2:vgg16; 3:resnet18
    samples_select_idx = 3  # 1:随机选择样本; 2:选择距离目标标签最远的样本; 3: 选择距离目标标签最近的样本
    model_name_dict = {1: 'AlexNet', 2: 'VGG16', 3: 'ResNet18', 5: 'VIT'}
    dataset_name_dict = {1: 'FashionMNIST', 2: 'CIFAR-10', 3: 'Tiny-ImageNet', 4: 'GTSRB'}
    # =======================================<<Parameter Modifying>>======================================= #
    for model_idx in model_idx_list:
        target_model, samples_dir, sel_dir = model_selection(model_idx, dataset_idx)
        quantized_model = torch.quantization.quantize_dynamic(target_model, {torch.nn.Linear, nn.Conv2d},
                                                              dtype=torch.qint8)
        quantized_model = quantized_model
        benign_model1 = benign_model_selection(model_idx, dataset_idx, 5, 1)
        benign_model2 = benign_model_selection(model_idx, dataset_idx, 0.2, 2)
        poisoned_model1 = poisoned_model_selection(model_idx, dataset_idx, 1, 1)
        poisoned_model2 = poisoned_model_selection(model_idx, dataset_idx, 10, 2)
        poisoned_model3 = poisoned_model_selection(model_idx, dataset_idx, 1, 3)
        poisoned_model4 = poisoned_model_selection(model_idx, dataset_idx, 2, 4)
        poisoned_model5 = poisoned_model_selection(model_idx, dataset_idx, 1, 5)
        poisoned_model6 = poisoned_model_selection(model_idx, dataset_idx, 1, 6)
        if dataset_idx == 1:
            data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                 transforms.Resize((224, 224)), transforms.ToTensor()])
        else:
            data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        if samples_idx == 1:
            dataset = ImageFolder(root=sel_dir, transform=data_transform)
            sample_name = 'original samples'
        else:
            dataset = ImageFolder(root=samples_dir, transform=data_transform)
            sample_name = 'semi-fragile samples'
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        target_model.eval()
        quantized_model.eval()
        benign_model1.eval()
        benign_model2.eval()
        poisoned_model1.eval()
        poisoned_model2.eval()
        poisoned_model3.eval()
        poisoned_model4.eval()
        poisoned_model5.eval()
        poisoned_model6.eval()
        with torch.no_grad():
            Vec1 = []
            Vec2 = []
            Vec3 = []
            Vec4 = []
            Vec5 = []
            Vec6 = []
            Vec7 = []
            Vec8 = []
            Vec9 = []
            Vec10 = []
            for images, labels in dataloader:
                # images = images.to(device)
                # labels = labels.to(device)
                # target_model = target_model.to(device)
                # quantized_model = quantized_model.to(device)
                # benign_model1 = benign_model1.to(device)
                # benign_model2 = benign_model2.to(device)
                # poisoned_model1 = poisoned_model1.to(device)
                # poisoned_model2 = poisoned_model2.to(device)
                # poisoned_model3 = poisoned_model3.to(device)
                # poisoned_model4 = poisoned_model4.to(device)
                if vector_idx == 1:
                    vector1 = target_model(images)
                    vector2 = quantized_model(images)
                    vector3 = benign_model1(images)
                    vector4 = benign_model2(images)
                    vector5 = poisoned_model1(images)
                    vector6 = poisoned_model2(images)
                    vector7 = poisoned_model3(images)
                    vector8 = poisoned_model4(images)
                    vector9 = poisoned_model5(images)
                    vector10 = poisoned_model6(images)
                    output_name = 'logit'
                else:
                    vector1 = torch.softmax(target_model(images), dim=1)
                    vector2 = torch.softmax(quantized_model(images), dim=1)
                    vector3 = torch.softmax(benign_model1(images), dim=1)
                    vector4 = torch.softmax(benign_model2(images), dim=1)
                    vector5 = torch.softmax(poisoned_model1(images), dim=1)
                    vector6 = torch.softmax(poisoned_model2(images), dim=1)
                    vector7 = torch.softmax(poisoned_model3(images), dim=1)
                    vector8 = torch.softmax(poisoned_model4(images), dim=1)
                    vector9 = torch.softmax(poisoned_model5(images), dim=1)
                    vector10 = torch.softmax(poisoned_model6(images), dim=1)
                    output_name = 'softmax'

                # Vec1.append(vector1[0].cpu().numpy())
                # Vec2.append(vector2[0].cpu().numpy())
                # Vec3.append(vector3[0].cpu().numpy())
                # Vec4.append(vector4[0].cpu().numpy())
                # Vec5.append(vector5[0].cpu().numpy())
                # Vec6.append(vector6[0].cpu().numpy())
                # Vec7.append(vector7[0].cpu().numpy())
                # Vec8.append(vector8[0].cpu().numpy())
                # Vec9.append(vector9[0].cpu().numpy())
                # Vec10.append(vector10[0].cpu().numpy())

                Vec1.append(vector1[0].numpy())
                Vec2.append(vector2[0].numpy())
                Vec3.append(vector3[0].numpy())
                Vec4.append(vector4[0].numpy())
                Vec5.append(vector5[0].numpy())
                Vec6.append(vector6[0].numpy())
                Vec7.append(vector7[0].numpy())
                Vec8.append(vector8[0].numpy())
                Vec9.append(vector9[0].numpy())
                Vec10.append(vector10[0].numpy())

            Vec1 = np.array(Vec1)
            Vec2 = np.array(Vec2)
            Vec3 = np.array(Vec3)
            Vec4 = np.array(Vec4)
            Vec5 = np.array(Vec5)
            Vec6 = np.array(Vec6)
            Vec7 = np.array(Vec7)
            Vec8 = np.array(Vec8)
            Vec9 = np.array(Vec9)
            Vec10 = np.array(Vec10)
            vectors = np.vstack((Vec1, Vec2, Vec3, Vec4, Vec5, Vec6, Vec7, Vec8, Vec9, Vec10))
            labels = np.array(
                [0] * Vec1.shape[0] + [1] * Vec1.shape[0] + [2] * Vec1.shape[0] + [3] * Vec1.shape[0] + [4] *
                Vec1.shape[0] + [5] * Vec1.shape[0] + [6] * Vec1.shape[0] + [7] * Vec1.shape[0] + [8] * Vec1.shape[
                    0] + [9] * Vec1.shape[0])
            # 使用t-SNE进行降维
            # 使用t-SNE进行降维
            n_runs = 1
            results = []
            for _ in range(n_runs):
                tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=26)
                embedded_vectors_tem = tsne.fit_transform(vectors)
                results.append(embedded_vectors_tem)
            embedded_vectors = np.mean(results, axis=0)
            # 为每个类别指定不同颜色
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'lime', 'tan', 'orange', 'k']
            n_classes = 10
            # 创建 Matplotlib 图形
            fig, ax = plt.subplots()
            # 绘制 t-SNE 可视化
            for class_id in range(n_classes):
                class_indices = labels == class_id
                ax.scatter(embedded_vectors[class_indices, 0], embedded_vectors[class_indices, 1],
                           c=colors[class_id], label=f'Class {class_id}')
            # 设置字体
            font = {'family': 'Times New Roman', 'style': 'italic', 'size': 12}
            plt.title(
                f"t-SNE: Visualization of {model_name_dict[model_idx]} \n {output_name} output of {sample_name} on dataset {dataset_name_dict[dataset_idx]}",
                fontdict=font)
            # ax.set_xlabel('t-SNE Dimension 1', fontdict=font)
            # ax.set_ylabel('t-SNE Dimension 2', fontdict=font)

            # 显示图例
            legend = ax.legend()
            for text in legend.get_texts():
                text.set_fontfamily('Times New Roman')
                text.set_fontsize(10)
                text.set_style('italic')
            plt.show()


if __name__ == '__main__':
    main()
