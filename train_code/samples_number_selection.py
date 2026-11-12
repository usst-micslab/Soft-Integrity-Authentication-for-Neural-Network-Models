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

# ================================<<环境设备设置>>================================ #
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/alexnet/"


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/vgg16/"

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/resnet18/"

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_fash/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_fash'
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/alexnet/"
            classifier_path = classifier_path + 'alexnet_cifar_fine-tuned_v1'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vgg16/"
            classifier_path = classifier_path + 'vgg16_cifar_fine-tuned_v1'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/resnet18/"
            classifier_path = classifier_path + 'resnet18_cifar_fine-tuned_v1'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_cifar/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/alexnet/"
            classifier_path = classifier_path + 'alexnet_tiny_fine-tuned_v1'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vgg16/"
            classifier_path = classifier_path + 'vgg16_tiny_fine-tuned_v1'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/resnet18/"
            classifier_path = classifier_path + 'resnet18_tiny_fine-tuned_v1'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_tiny/mobilenet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/mobilenet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
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
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"
            classifier_path = classifier_path + 'alexnet_gtsrb_fine-tuned_v1'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/vgg16/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/vgg16/"
            classifier_path = classifier_path + 'vgg16_gtsrb_fine-tuned_v1'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/resnet18/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/resnet18/"
            classifier_path = classifier_path + 'resnet18_gtsrb_fine-tuned_v1'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            samples_dir = "./semi-fragile_samples_gtsrb/vit/"
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/vit/"
            classifier_path = classifier_path + 'vit_gtsrb_fine-tuned_v1'

    return model, samples_dir, sel_dir, classifier_path


def main():
    # =====================<<使用投票机制来识别对于每种修改类型>>===================== #
    dataset_idx_list = [2, 3, 4]  # 1: FashionMNST; 2: CIFAR-10; 3: Tiny-ImageNet; 4: GTSRB
    model_idx_list = [1, 2, 3, 5]  # 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT
    # samples_num_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    samples_num_list = [10, 20, 30]
    for dataset_idx in dataset_idx_list:
        for model_idx in model_idx_list:
            target_model, samples_dir, sel_dir, classifier_path = model_selection(model_idx, dataset_idx)
            prediction_results = np.load(samples_dir + 'cla_results.npy')
            for samples_num in samples_num_list:
                c1 = 0
                c2 = 0
                c3 = 0
                c4 = 0
                c5 = 0
                c6 = 0
                c7 = 0
                c8 = 0
                c9 = 0
                c10 = 0
                for k in range(1000):
                    index = np.random.choice(100, samples_num, replace=False)
                    ori_pre = prediction_results[:1, index]
                    qua_pre = prediction_results[1:2, index]
                    fin_pre = prediction_results[2:12, index]
                    pru_pre = prediction_results[12:14, index]
                    bad_pre = prediction_results[14:24, index]
                    wan_pre = prediction_results[24:34, index]
                    ble_pre = prediction_results[34:44, index]
                    iss_pre = prediction_results[44:54, index]
                    iba_pre = prediction_results[54:64, index]
                    lc_pre = prediction_results[64:74, index]
                    ori_pre_label = np.bincount(np.squeeze(ori_pre.astype(int))).argmax()
                    if ori_pre_label == 0:
                        c1 += 1

                    qua_pre_label = np.bincount(np.squeeze(qua_pre.astype(int))).argmax()
                    if qua_pre_label == 1:
                        c2 += 1

                    for i in range(10):
                        tem_pre = fin_pre[i, :]
                        fin_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if fin_pre_label == 2:
                            c3 += 1

                    for i in range(2):
                        tem_pre = pru_pre[i, :]
                        pru_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if pru_pre_label == 3:
                            c4 += 1

                    for i in range(10):
                        tem_pre = bad_pre[i, :]
                        bad_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if bad_pre_label == 4:
                            c5 += 1

                    for i in range(10):
                        tem_pre = wan_pre[i, :]
                        wan_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if wan_pre_label == 5:
                            c6 += 1

                    for i in range(10):
                        tem_pre = ble_pre[i, :]
                        wan_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if wan_pre_label == 6:
                            c7 += 1

                    for i in range(10):
                        tem_pre = iss_pre[i, :]
                        iss_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if iss_pre_label == 7:
                            c8 += 1

                    for i in range(10):
                        tem_pre = iba_pre[i, :]
                        iba_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if iba_pre_label == 8:
                            c9 += 1

                    for i in range(10):
                        tem_pre = lc_pre[i, :]
                        lc_pre_label = np.bincount(np.squeeze(tem_pre.astype(int))).argmax()
                        if lc_pre_label == 9:
                            c10 += 1

                print(
                    f'# =====The dataset index is: {dataset_idx}, | The model index is: {model_idx}, | the samples number: {samples_num}===== #')
                print('The accuracy of MTI: Original: {}, | Quan: {}, Fine-tuning: {}, | Pruning: {}'.format(c1 / 1000,
                                                                                                             c2 / 1000,
                                                                                                             c3 / 10000,
                                                                                                             c4 / 2000))
                print(
                    'The accuracy of MTI: BadNet: {}, | WaNet: {}, Blended: {}, | ISSBA: {}, IBA: {}, | LC: {}'.format(
                        c5 / 10000, c6 / 10000, c7 / 10000, c8 / 10000, c9 / 10000, c10 / 10000))
                print(
                    f'# =====The dataset index is: {dataset_idx}, | The model index is: {model_idx}, | the samples number: {samples_num}===== #')
                print('\n')


if __name__ == '__main__':
    main()
