import os
import math
import timm
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from adversarial_sample_generation import *
import torchvision.transforms as transforms

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg)

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
            samples_dir = "./semi-fragile_samples_fash/alexnet/"


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/resnet18/"

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/resnet18/"

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/mobilenet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            elif poisoned_idx == 5:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/vit_gtsrb_ifsr_' \
                             + str(pruning_idx)
            elif poisoned_idx == 6:
                model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/gtsrb/vit_gtsrb_LC_' \
                             + str(pruning_idx)
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    return model


def acc_test(test_loader, model, target_label):
    model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            # images = images.to(device)
            # labels = labels.to(device)
            # model = model.to(device)
            new_labels = torch.ones_like(labels) * target_label
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == new_labels).sum().item()
            acc = 100 * correct / len(test_loader)
        print('The classification accuracy of model:{}'.format(acc))
    return 0


batch_size = 1
original_label = 0
target_label = 3
# =======================================<<Parameter Modifying>>======================================= #
dataset_idx = 2  # 1: FashionMNST; 2: CIFAR-10; 3: Tiny-ImageNet; 4: GTSRB
model_idx_list = [5]  # 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT
# samples_select_idx = 3  # 1:samples random; 2:farthest; 3: shortest
fintuning_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
pruning_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
backdoor_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# =======================================<<Parameter Modifying>>======================================= #
for model_idx in model_idx_list:
    target_model, samples_dir = model_selection(model_idx, dataset_idx)
    # quantized_model = torch.quantization.quantize_dynamic(target_model, {torch.nn.Linear, nn.Conv2d},
    #                                                       dtype=torch.qint8)
    quantized_model = torch.quantization.quantize_dynamic(target_model, {torch.nn.Linear},
                                                          dtype=torch.qint8)
    quantized_model = quantized_model
    if dataset_idx == 1:
        data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                             transforms.Resize((224, 224)), transforms.ToTensor()])
    else:
        data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(root=samples_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    acc_test(dataloader, target_model, target_label)
    acc_test(dataloader, quantized_model, target_label)
    print('\n')
    print('# =======================<<Fintuned Model>>======================= #')
    for fintuning_idx in fintuning_list:
        print('The model index: {}, | the fintuning index: {}'.format(model_idx, fintuning_idx))
        benign_idx = 1
        benign_model = benign_model_selection(model_idx, dataset_idx, fintuning_idx, benign_idx)
        acc_test(dataloader, benign_model, target_label)
        print('\n')

    print('# =======================<<Pruning Model>>======================= #')
    for pruning_idx in pruning_list:
        print('The model index: {}, | the pruning index: {}'.format(model_idx, pruning_idx))
        benign_idx = 2
        benign_model = benign_model_selection(model_idx, dataset_idx, pruning_idx, benign_idx)
        acc_test(dataloader, benign_model, target_label)
        print('\n')

    print('# =======================<<Poisoned Model>>======================= #')
    for backdoor_idx in backdoor_list:
        print('The model index: {}, | the BadNet attack index: {}'.format(model_idx, backdoor_idx))
        poisoned_idx = 1
        poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
        acc_test(dataloader, poisoned_model, target_label)
        print('The model index: {}, | the WaNet attack index: {}'.format(model_idx, backdoor_idx))
        poisoned_idx = 2
        poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
        acc_test(dataloader, poisoned_model, target_label)
        print('The model index: {}, | the Blended attack index: {}'.format(model_idx, backdoor_idx))
        poisoned_idx = 3
        poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
        acc_test(dataloader, poisoned_model, target_label)
        print('The model index: {}, | the ISSBA attack index: {}'.format(model_idx, backdoor_idx))
        poisoned_idx = 4
        poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
        acc_test(dataloader, poisoned_model, target_label)
        print('The model index: {}, | the IFSR attack index: {}'.format(model_idx, backdoor_idx))
        poisoned_idx = 5
        poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
        acc_test(dataloader, poisoned_model, target_label)
        print('The model index: {}, | the LC attack index: {}'.format(model_idx, backdoor_idx))
        poisoned_idx = 6
        poisoned_model = poisoned_model_selection(model_idx, dataset_idx, backdoor_idx, poisoned_idx)
        acc_test(dataloader, poisoned_model, target_label)
        print('\n')

