from experiment_config import checkpoint, load_classifier, load_model_state
import os
import math
import time
import timm
import copy
import torch
import random
import warnings
import numpy as np
import torchvision
import torch.nn as nn
import os.path as osp
# from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from adversarial_sample_generation import *
import torchvision.transforms as transforms
from Tinyimagenet_dataset import TinyImageNet
from pytorch_msssim import ssim

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
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        hidden_vec = self.layers(x)
        output = self.fc(hidden_vec)
        return output


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def model_selection(model_idx, dataset_idx, target_label):
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
            samples_dir = "./semi-fragile_samples_fash/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/alexnet/"
            classifier_model = load_classifier('./classifier/Map_alexnet_fash.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.3914, 0.9359, -2.44, 5.6519

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/vgg16/"
            classifier_model = load_classifier('./classifier/Map_vgg16_fash.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.0364, 0.7443, -3.1100, 6.9168

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/resnet18/"
            classifier_model = load_classifier('./classifier/Map_rsnet18_fash.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.3081, 1.2404, -1.2378, 5.6998

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/mobilenet/"
            ben_mean, ben_std, poi_mean, poi_std = -0.3914, 0.9359, -2.44, 5.6519

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = checkpoint("test_model/vit_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_fash/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_fash/vit/"
            classifier_model = load_classifier('./classifier/Map_vit_fash.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = 0.0207, 1.8286, -0.0326, 5.6369

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
            samples_dir = "./semi-fragile_samples_cifar/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/alexnet/"
            classifier_model = load_classifier('./classifier/Map_alexnet_cifar_v1.pth', ProjectionHead)

            ben_mean, ben_std, poi_mean, poi_std = -0.1895, 0.9564, -1.6860, 4.8211

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vgg16/"
            classifier_model = load_classifier('./classifier/Map_vgg16_cifar.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.1168, 0.5519, -2.0174, 4.6614

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/resnet18/"
            classifier_model = load_classifier('./classifier/Map_resnet18_cifar.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.0506, 0.4856, -1.1437, 5.2260

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/mobilenet/"
            ben_mean, ben_std, poi_mean, poi_std = -0.3914, 0.9359, -2.44, 5.6519

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_cifar/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_cifar/vit/"
            classifier_model = load_classifier('./classifier/Map_vit_cifar_v1.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.0227, 1.0126, -0.2320, 3.7705

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
            samples_dir = "./semi-fragile_samples_tiny/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/alexnet/"
            classifier_model = load_classifier('./classifier/Map_alexnet_tiny.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -2.4868, 1.2726, -8.0833, 9.7206

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vgg16/"
            classifier_model = load_classifier('./classifier/Map_vgg16_tiny.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -2.9770, 1.6236, -8.7291, 7.8321

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/resnet18/"
            classifier_model = load_classifier('./classifier/Map_resnet18_tiny.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -1.3394, 1.3599, 0.5469, 2.5276

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/mobilenet/"
            ben_mean, ben_std, poi_mean, poi_std = -0.3914, 0.9359, -2.44, 5.6519

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_tiny/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_tiny/vit/"
            classifier_model = load_classifier('./classifier/Map_vit_tiny_v1.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.0163, 0.4341, 0.0103, 1.4657

    elif dataset_idx == 4:
        print('# =====================<<Dataset is GTSRB>>===================== #')
        num_classes = 43
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/alexnet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/alexnet/"
            classifier_model = load_classifier('./classifier/Map_alexnet_gtsrb.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = 0.0380, 1.9016, -0.0461, 4.1411

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/vgg16/"
            classifier_model = load_classifier('./classifier/Map_vgg16_gtsrb.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = 0.4600, 9.0046, 0.1224, 6.6630

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/resnet18/"
            classifier_model = load_classifier('./classifier/Map_resnet18_gtsrb.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = 0.6276, 2.9422, -0.1990, 5.0769

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/mobilenet/"
            ben_mean, ben_std, poi_mean, poi_std = -0.3914, 0.9359, -2.44, 5.6519

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./semi-fragile_samples_gtsrb/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)
            sel_dir = "./selected_samples_gtsrb/vit/"
            classifier_model = load_classifier('./classifier/Map_vit_gtsrb_v1.pth', ProjectionHead)
            ben_mean, ben_std, poi_mean, poi_std = -0.0150, 0.5513, -0.0312, 3.3246

    return model, samples_dir, sel_dir, num_classes, ben_mean, ben_std, poi_mean, poi_std, classifier_model


def benign_model_selection(model_idx, dataset_idx):
    if dataset_idx == 1:
        print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/alexnet_fash_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/vgg16_fash_50")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/resnet18_fash_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/mobilenet_fash_tiny")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = checkpoint("fintuned_model/VIT_fash_30")
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 2:
        print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/alexnet_cifar_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/vgg16_cifar_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/resnet18_cifar_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/mobilenet_cifar_tiny")
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("fintuned_model/VIT_cifar_30")
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 3:
        print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        tem_path = r'G:\SIANN\fintuned_model/'
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\fintuned_model/alexnet_tiny_30'
            model_path = tem_path + 'alexnet_tiny_30'
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\fintuned_model/vgg16_tiny_30'
            model_path = tem_path + 'vgg16_tiny_30'
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\fintuned_model/resnet18_tiny_50'
            model_path = tem_path + 'resnet18_tiny_50'
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/mobilenet_tiny_tiny")
            model_path = tem_path + 'resnet18_tiny_50'
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\fintuned_model/VIT_tiny_30'
            model_path = tem_path + 'VIT_tiny_30'
            model.load_state_dict(load_model_state(model_path))

    elif dataset_idx == 4:
        print('# =====================<<Dataset is GTSRB>>===================== #')
        num_classes = 43
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/alexnet_gtsrb_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/vgg16_gtsrb_10")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/resnet18_gtsrb_30")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("fintuned_model/mobilenet_gtsrb_tiny")
            model.load_state_dict(load_model_state(model_path))

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("fintuned_model/VIT_gtsrb_30")
            model.load_state_dict(load_model_state(model_path))

    return model


def poisoned_model_selection(model_idx, dataset_idx):
    if dataset_idx == 1:
        print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model1 = torchvision.models.alexnet(pretrained=False)
            model1.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/alexnet/fash/alexnet_fash_badnet_5")
            model2 = torchvision.models.alexnet(pretrained=False)
            model2.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/alexnet/fash/alexnet_fash_WaNet_5")
            model3 = torchvision.models.alexnet(pretrained=False)
            model3.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/alexnet/fash/alexnet_fash_blended_2")
            model1.load_state_dict(load_model_state(model_path1))
            model2.load_state_dict(load_model_state(model_path2))
            model3.load_state_dict(load_model_state(model_path3))

        elif model_idx == 2:
            model1 = torchvision.models.vgg16(pretrained=False)
            model1.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/vgg16/fash/vgg16_fash_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.vgg16(pretrained=False)
            model2.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/vgg16/fash/vgg16_fash_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.vgg16(pretrained=False)
            model3.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/vgg16/fash/vgg16_fash_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

        elif model_idx == 3:
            model1 = torchvision.models.resnet18(pretrained=False)
            model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model1.fc.in_features  # 获取最后一个全连接层的输入是多少
            model1.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/resnet18/fash/resnet18_fash_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.resnet18(pretrained=False)
            model2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model2.fc.in_features  # 获取最后一个全连接层的输入是多少
            model2.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/resnet18/fash/resnet18_fash_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.resnet18(pretrained=False)
            model3.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model3.fc.in_features  # 获取最后一个全连接层的输入是多少
            model3.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/resnet18/fash/resnet18_fash_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("backdoor_model/alexnet/fash/alexnet_fash_badnet_")
            model.load_state_dict(load_model_state(model_path))

        else:
            model1 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model1.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path1 = checkpoint("backdoor_model/vit/fash/VIT_fash_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model2.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path2 = checkpoint("backdoor_model/vit/fash/VIT_fash_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model3.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path3 = checkpoint("backdoor_model/vit/fash/VIT_fash_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

    elif dataset_idx == 2:
        print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model1 = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_blended_4")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model4.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path4 = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_issba_5")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model5.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model5.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path5 = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model6.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model6.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/alexnet/cifar/alexnet_cifar_LC_5")
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 2:
            model1 = torchvision.models.vgg16(pretrained=False)
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.vgg16(pretrained=False)
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.vgg16(pretrained=False)
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.vgg16(pretrained=False)
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model4.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path4 = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_issba_5")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.vgg16(pretrained=False)
            num_fc = model5.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model5.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path5 = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.vgg16(pretrained=False)
            num_fc = model6.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model6.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/vgg16/cifar/vgg16_cifar_LC_5")
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 3:
            model1 = torchvision.models.resnet18(pretrained=False)
            num_fc = model1.fc.in_features  # 获取最后一个全连接层的输入是多少
            model1.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.resnet18(pretrained=False)
            num_fc = model2.fc.in_features  # 获取最后一个全连接层的输入是多少
            model2.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.resnet18(pretrained=False)
            num_fc = model3.fc.in_features  # 获取最后一个全连接层的输入是多少
            model3.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.resnet18(pretrained=False)
            num_fc = model4.fc.in_features  # 获取最后一个全连接层的输入是多少
            model4.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path4 = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_issba_5")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.resnet18(pretrained=False)
            num_fc = model5.fc.in_features  # 获取最后一个全连接层的输入是多少
            model5.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path5 = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.resnet18(pretrained=False)
            num_fc = model6.fc.in_features  # 获取最后一个全连接层的输入是多少
            model6.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/resnet18/cifar/resnet18_cifar_LC_5")
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("backdoor_model/vit/cifar/VIT_badnet_cifar_5")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 5:
            model1 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path1 = checkpoint("backdoor_model/vit/cifar/VIT_cifar_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path2 = checkpoint("backdoor_model/vit/cifar/VIT_cifar_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path3 = checkpoint("backdoor_model/vit/cifar/VIT_cifar_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path4 = checkpoint("backdoor_model/vit/cifar/vit_cifar_issba_5")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path5 = checkpoint("backdoor_model/vit/cifar/vit_cifar_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path6 = checkpoint("backdoor_model/vit/cifar/vit_cifar_LC_1")
            model6.load_state_dict(load_model_state(model_path6))

    elif dataset_idx == 3:
        # print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            tem_path = r'G:\SIANN\backdoor_model\alexnet\tiny/'
            model1 = torchvision.models.alexnet(pretrained=False)
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path1 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_badnet_5'
            model_path1 = tem_path + 'alexnet_tiny_badnet_5'
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.alexnet(pretrained=False)
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path2 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_WaNet_5'
            model_path2 = tem_path + 'alexnet_tiny_WaNet_5'
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.alexnet(pretrained=False)
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path3 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_blended_5'
            model_path3 = tem_path + 'alexnet_tiny_blended_5'
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.alexnet(pretrained=False)
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model4.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path4 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_issba_5'
            model_path4 = tem_path + 'alexnet_tiny_issba_5'
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.alexnet(pretrained=False)
            num_fc = model5.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model5.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path5 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_ifsr_5'
            model_path5 = tem_path + 'alexnet_tiny_ifsr_5'
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.alexnet(pretrained=False)
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model6.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path6 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/alexnet/tiny/alexnet_tiny_LC_5'
            model_path6 = tem_path + 'alexnet_tiny_LC_5'
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 2:
            model1 = torchvision.models.vgg16(pretrained=False)
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path1 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_badnet_3'
            model_path1 = r'G:\SIANN\backdoor_model\vgg16\tiny/vgg16_tiny_badnet_3'
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.vgg16(pretrained=False)
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path2 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_WaNet_5'
            model_path2 = r'G:\SIANN\backdoor_model\vgg16\tiny/vgg16_tiny_WaNet_5'
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.vgg16(pretrained=False)
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path3 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_blended_1'
            model_path3 = r'G:\SIANN\backdoor_model\vgg16\tiny/vgg16_tiny_blended_1'
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.vgg16(pretrained=False)
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model4.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path4 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_issba_5'
            model_path4 = r'G:\SIANN\backdoor_model\vgg16\tiny/vgg16_tiny_issba_5'
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.vgg16(pretrained=False)
            num_fc = model5.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model5.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path5 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vgg16/tiny/vgg16_tiny_ifsr_5'
            model_path5 = r'G:\SIANN\backdoor_model\vgg16\tiny/vgg16_tiny_ifsr_5'
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.vgg16(pretrained=False)
            num_fc = model6.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model6.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/vgg16/tiny/vgg16_tiny_LC_5")
            model_path6 = r'G:\SIANN\backdoor_model\vgg16\tiny/vgg16_tiny_LC_5'
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 3:
            tem_path = r'G:\SIANN\backdoor_model\resnet18\tiny/'
            model1 = torchvision.models.resnet18(pretrained=False)
            num_fc = model1.fc.in_features  # 获取最后一个全连接层的输入是多少
            model1.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path1 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_badnet_5'
            model_path1 = tem_path + 'resnet18_tiny_badnet_5'
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.resnet18(pretrained=False)
            num_fc = model2.fc.in_features  # 获取最后一个全连接层的输入是多少
            model2.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path2 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_WaNet_5'
            model_path2 = tem_path + 'resnet18_tiny_WaNet_5'
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.resnet18(pretrained=False)
            num_fc = model3.fc.in_features  # 获取最后一个全连接层的输入是多少
            model3.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path3 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_blended_5'
            model_path3 = tem_path + 'resnet18_tiny_blended_5'
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.resnet18(pretrained=False)
            num_fc = model4.fc.in_features  # 获取最后一个全连接层的输入是多少
            model4.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path4 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_issba_5'
            model_path4 = tem_path + 'resnet18_tiny_issba_5'
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.resnet18(pretrained=False)
            num_fc = model5.fc.in_features  # 获取最后一个全连接层的输入是多少
            model5.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path5 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_ifsr_5'
            model_path5 = tem_path + 'resnet18_tiny_ifsr_5'
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.resnet18(pretrained=False)
            num_fc = model6.fc.in_features  # 获取最后一个全连接层的输入是多少
            model6.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path6 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/resnet18/tiny/resnet18_tiny_LC_5'
            model_path6 = tem_path + 'resnet18_tiny_LC_5'
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("backdoor_model/alexnet/tiny/alexnet_tiny_badnet_")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 5:
            tem_path = r'G:\SIANN\backdoor_model\vit\tiny/'
            model1 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path1 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/VIT_tiny_badnet_5'
            model_path1 = tem_path + 'VIT_tiny_badnet_5'
            model1.load_state_dict(load_model_state(model_path1))

            model2 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path2 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/VIT_tiny_WaNet_5'
            model_path2 = tem_path + 'VIT_tiny_WaNet_5'
            model2.load_state_dict(load_model_state(model_path2))

            model3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path3 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/VIT_tiny_blended_5'
            model_path3 = tem_path + 'VIT_tiny_blended_5'
            model3.load_state_dict(load_model_state(model_path3))

            model4 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path4 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/vit_tiny_issba_5'
            model_path4 = tem_path + 'vit_tiny_issba_5'
            model4.load_state_dict(load_model_state(model_path4))

            model5 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path5 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/vit_tiny_ifsr_5'
            model_path5 = tem_path + 'vit_tiny_ifsr_5'
            model5.load_state_dict(load_model_state(model_path5))

            model6 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            # model_path6 = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\backdoor_model/vit/tiny/vit_tiny_LC_5'
            model_path6 = tem_path + 'vit_tiny_LC_5'
            model6.load_state_dict(load_model_state(model_path6))

    elif dataset_idx == 4:
        # print('# =====================<<Dataset is Tiny GTSRB>>===================== #')
        num_classes = 43
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model1 = torchvision.models.alexnet(pretrained=False)
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.alexnet(pretrained=False)
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.alexnet(pretrained=False)
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_blended_1")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.alexnet(pretrained=False)
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model4.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path4 = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_issba_6")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.alexnet(pretrained=False)
            num_fc = model5.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model5.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path5 = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.alexnet(pretrained=False)
            num_fc = model6.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model6.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_LC_5")
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 2:
            model1 = torchvision.models.vgg16(pretrained=False)
            num_fc = model1.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model1.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_badnet_3")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.vgg16(pretrained=False)
            num_fc = model2.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model2.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_WaNet_1")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.vgg16(pretrained=False)
            num_fc = model3.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_blended_7")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.vgg16(pretrained=False)
            num_fc = model4.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model4.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path4 = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_issba_5")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.vgg16(pretrained=False)
            num_fc = model5.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model5.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path5 = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.vgg16(pretrained=False)
            num_fc = model6.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model6.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/vgg16/gtsrb/vgg16_gtsrb_LC_2")
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 3:
            model1 = torchvision.models.resnet18(pretrained=False)
            num_fc = model1.fc.in_features  # 获取最后一个全连接层的输入是多少
            model1.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path1 = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = torchvision.models.resnet18(pretrained=False)
            num_fc = model2.fc.in_features  # 获取最后一个全连接层的输入是多少
            model2.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path2 = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = torchvision.models.resnet18(pretrained=False)
            num_fc = model3.fc.in_features  # 获取最后一个全连接层的输入是多少
            model3.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path3 = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_blended_5")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = torchvision.models.resnet18(pretrained=False)
            num_fc = model4.fc.in_features  # 获取最后一个全连接层的输入是多少
            model4.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path4 = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_issba_5")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = torchvision.models.resnet18(pretrained=False)
            num_fc = model5.fc.in_features  # 获取最后一个全连接层的输入是多少
            model5.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path5 = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_ifsr_5")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = torchvision.models.resnet18(pretrained=False)
            num_fc = model6.fc.in_features  # 获取最后一个全连接层的输入是多少
            model6.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path6 = checkpoint("backdoor_model/resnet18/gtsrb/resnet18_gtsrb_LC_5")
            model6.load_state_dict(load_model_state(model_path6))

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("backdoor_model/alexnet/gtsrb/alexnet_gtsrb_badnet_")
            model.load_state_dict(load_model_state(model_path))

        elif model_idx == 5:
            model1 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path1 = checkpoint("backdoor_model/vit/gtsrb/VIT_gtsrb_badnet_5")
            model1.load_state_dict(load_model_state(model_path1))

            model2 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path2 = checkpoint("backdoor_model/vit/gtsrb/VIT_gtsrb_WaNet_5")
            model2.load_state_dict(load_model_state(model_path2))

            model3 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path3 = checkpoint("backdoor_model/vit/gtsrb/VIT_gtsrb_blended_1")
            model3.load_state_dict(load_model_state(model_path3))

            model4 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path4 = checkpoint("backdoor_model/vit/gtsrb/vit_gtsrb_issba_1")
            model4.load_state_dict(load_model_state(model_path4))

            model5 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path5 = checkpoint("backdoor_model/vit/gtsrb/vit_gtsrb_ifsr_1")
            model5.load_state_dict(load_model_state(model_path5))

            model6 = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path6 = checkpoint("backdoor_model/vit/gtsrb/vit_gtsrb_LC_1")
            model6.load_state_dict(load_model_state(model_path6))

    return model1, model2, model3, model4, model5, model6


# def ssim(img1, img2, window_size=11, size_average=True):
#     img1 = img1.cpu()
#     img2 = img2.cpu()
#     window = create_window(window_size, img1.size(1))
#
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img1.size(1))
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
#
#     C1 = (0.0001 * 255) ** 2  # 通常为0.01
#     C2 = (0.0003 * 255) ** 2  # 通常为0.03
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# def create_window(window_size, channel):
#     sigma = 1.5
#     gauss = torch.Tensor(
#         [torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))) for x in range(window_size)])
#     window = torch.outer(gauss, gauss)
#     window = window.view(1, 1, window_size, window_size)
#     window = window.expand(channel, 1, window_size, window_size).contiguous()
#     return window / window.sum()
#
#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor(
#         [torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))) for x in range(window_size)])
#     return gauss / gauss.sum()


def train(target_model, benign_model, poisoned_model1, poisoned_model2, poisoned_model3, poisoned_model4,
          poisoned_model5, poisoned_model6, dataset, semi_fragile_samples_dir, alfa_1, alfa_2, alfa_3, alfa_4,
          epoch, sample_lr, original_label, target_label, num_classes, ben_mean, ben_std, poi_mean, poi_std,
          classifier_model, path3, noise_std1, noise_std2):
    # ==================================<<loss function>>================================== #
    train_start_time = time.time()
    CE_loss = nn.CrossEntropyLoss()
    KL_loss = nn.KLDivLoss()
    L2_loss = nn.MSELoss()
    triplet_loss = torch.nn.TripletMarginLoss(reduction='mean')
    cos_loss = nn.CosineSimilarity(dim=0)
    cosine_similarity_loss = nn.CosineEmbeddingLoss(margin=0.2, reduction="mean")
    loss_cts = ContrastiveLoss(batch_size=4)
    # ==================================<<loss function>>================================== #
    target_model = target_model.to(device)
    benign_model = benign_model.to(device)
    poisoned_model1 = poisoned_model1.to(device)
    poisoned_model2 = poisoned_model2.to(device)
    poisoned_model3 = poisoned_model3.to(device)
    poisoned_model4 = poisoned_model4.to(device)
    poisoned_model5 = poisoned_model5.to(device)
    poisoned_model6 = poisoned_model6.to(device)
    semi_fragile_samples = []
    img_idx = 0
    MSE_tem = 0
    SSIM_tem1 = 0
    sgd = optim.SGD(classifier_model.parameters(), lr=0.001, momentum=0.9)
    for imgs, labels in dataset:
        classifier_model = classifier_model.to(device)
        imgs = imgs.to(device)
        labels = labels.to(device)
        new_labels = torch.ones_like(labels) * target_label
        error_label = torch.ones_like(labels) * original_label
        imgs_backup = copy.deepcopy(imgs)
        adv_images = imgs.clone().detach()
        for epoch_idx in range(epoch):
            adv_images.requires_grad = True
            logit_ori_vector = target_model(adv_images)  # 原始模型的逻辑输出
            logit_ori_vector_beg = benign_model(adv_images)  # 良性模型的逻辑输出
            logit_poi_vector1 = poisoned_model1(adv_images)
            logit_poi_vector2 = poisoned_model2(adv_images)
            logit_poi_vector3 = poisoned_model3(adv_images)
            logit_poi_vector4 = poisoned_model4(adv_images)
            logit_poi_vector5 = poisoned_model5(adv_images)
            logit_poi_vector6 = poisoned_model6(adv_images)

            noise_beg = torch.randn_like(logit_ori_vector_beg) * 1 + 0  # 为良性模型输出生成噪声
            noise_poi1 = torch.randn_like(logit_poi_vector1) * 1 + 0
            noise_poi2 = torch.randn_like(logit_poi_vector2) * 1 + 0
            noise_poi3 = torch.randn_like(logit_poi_vector3) * 1 + 0
            noise_poi4 = torch.randn_like(logit_poi_vector4) * 1 + 0
            noise_poi5 = torch.randn_like(logit_poi_vector5) * 1 + 0
            noise_poi6 = torch.randn_like(logit_poi_vector6) * 1 + 0

            noisy_ori_vector = logit_ori_vector_beg + noise_beg
            noisy_poi_vector1 = logit_poi_vector1 + noise_poi1
            noisy_poi_vector2 = logit_poi_vector2 + noise_poi2
            noisy_poi_vector3 = logit_poi_vector3 + noise_poi3
            noisy_poi_vector4 = logit_poi_vector4 + noise_poi4
            noisy_poi_vector5 = logit_poi_vector5 + noise_poi5
            noisy_poi_vector6 = logit_poi_vector6 + noise_poi6

            anchor_vector = F.normalize(logit_ori_vector, p=2, dim=1)
            benign_vector = F.normalize(noisy_ori_vector, p=2, dim=1)
            poisoned_vector1 = F.normalize(noisy_poi_vector1, p=2, dim=1)
            poisoned_vector2 = F.normalize(noisy_poi_vector2, p=2, dim=1)
            poisoned_vector3 = F.normalize(noisy_poi_vector3, p=2, dim=1)
            poisoned_vector4 = F.normalize(noisy_poi_vector4, p=2, dim=1)
            poisoned_vector5 = F.normalize(noisy_poi_vector5, p=2, dim=1)
            poisoned_vector6 = F.normalize(noisy_poi_vector6, p=2, dim=1)

            loss_ce_ori = CE_loss(logit_ori_vector, new_labels)  # 原始模型将半脆弱样本识别为目标标签
            loss_ce_ben = CE_loss(noisy_ori_vector, new_labels)  # 原始模型将半脆弱样本识别为目标标签
            loss_ce_poi1 = CE_loss(noisy_poi_vector1, error_label)
            loss_ce_poi2 = CE_loss(noisy_poi_vector2, error_label)
            loss_ce_poi3 = CE_loss(noisy_poi_vector3, error_label)
            loss_ce_poi4 = CE_loss(noisy_poi_vector4, error_label)
            loss_ce_poi5 = CE_loss(noisy_poi_vector5, error_label)
            loss_ce_poi6 = CE_loss(noisy_poi_vector6, error_label)

            # ======================<<classifier_model_vec>>====================== #
            vec = torch.vstack((logit_ori_vector, noisy_ori_vector, noisy_poi_vector1, noisy_poi_vector2,
                                noisy_poi_vector3, noisy_poi_vector4, noisy_poi_vector5, noisy_poi_vector6))
            cla_label = torch.ones(8)
            cla_label[0] = 0
            cla_label[1] = 2
            cla_label[2] = 4
            cla_label[3] = 5
            cla_label[4] = 6
            cla_label[5] = 7
            cla_label[6] = 8
            cla_label[7] = 9
            cla_label = cla_label.to(device)
            loss_cla = CE_loss(classifier_model(vec), cla_label.long())
            # # ====================================<<Method_1>>==================================== #
            # # ===========<<使用KL散度来拉近良性输出的距离，扩大中毒输出的距离>>=========== #
            # # KL_large = KL_loss(torch.log(original_vector), poisoned_vector)
            KL_loss1 = triplet_loss(anchor_vector, benign_vector, poisoned_vector1)
            KL_loss2 = triplet_loss(anchor_vector, benign_vector, poisoned_vector2)
            KL_loss3 = triplet_loss(anchor_vector, benign_vector, poisoned_vector3)
            KL_loss4 = triplet_loss(anchor_vector, benign_vector, poisoned_vector4)
            KL_loss5 = triplet_loss(anchor_vector, benign_vector, poisoned_vector5)
            KL_loss6 = triplet_loss(anchor_vector, benign_vector, poisoned_vector6)
            loss_ce_ben = loss_ce_ori + loss_ce_ben
            loss_ce_poi = loss_ce_poi1 + loss_ce_poi2 + loss_ce_poi3 + loss_ce_poi4 + loss_ce_poi5 + loss_ce_poi6
            loss_ce = alfa_1 * loss_ce_ben + alfa_2 * loss_ce_poi
            loss_tri = KL_loss1 + KL_loss2 + KL_loss3 + KL_loss4 + KL_loss5 + KL_loss6
            if epoch_idx < 150:
                total_loss = loss_ce
            else:
                total_loss = loss_ce + alfa_3 * loss_tri + alfa_4 * loss_cla
            print('Image_idx\epoch_idx: {}\{}'.format(img_idx, epoch_idx))
            print('The CE loss of Classifier model: {:.5f}'.format(loss_cla))
            print('The CE loss of Ori and Ben: {:.5f}, | {:.5f}'.format(loss_ce_ori, loss_ce_ben))
            print('The CE loss of Pois: {:.5f}, | {:.5f}, | {:.5f}, | {:.5f}, | {:.5f}, | {:.5f}'.format(loss_ce_poi1,
                                                                                                         loss_ce_poi2,
                                                                                                         loss_ce_poi3,
                                                                                                         loss_ce_poi4,
                                                                                                         loss_ce_poi5,
                                                                                                         loss_ce_poi6))
            print('The tri losses: {:.5f}, | {:.5f}, | {:.5f}, | {:.5f}, | {:.5f}, | {:.5f}'.format(KL_loss1, KL_loss2,
                                                                                                    KL_loss3, KL_loss4,
                                                                                                    KL_loss5, KL_loss6))
            print('The total loss:', total_loss.item())
            print('\n')
            target_model.zero_grad()
            benign_model.zero_grad()
            poisoned_model1.zero_grad()
            poisoned_model2.zero_grad()
            poisoned_model3.zero_grad()
            poisoned_model4.zero_grad()
            poisoned_model5.zero_grad()
            poisoned_model6.zero_grad()
            total_loss.backward(retain_graph=True)
            with torch.no_grad():
                adv_sample_grad = adv_images.grad.data
                adv_images = torch.clamp(adv_images - sample_lr * adv_sample_grad, 0, 1)
            sgd.zero_grad()
            loss_cla.backward()
            sgd.step()

        img_idx += 1
        MSE_t = L2_loss(adv_images, imgs_backup).item()
        print('The MSE between original image and semi-image:', )
        MSE_tem += MSE_t
        PSNR_tem = 20 * math.log10(1.0 / math.sqrt(L2_loss(adv_images, imgs_backup).item()))
        print('The PSNR between original image and semi-image:', PSNR_tem)
        # SSIM_tem = ssim(adv_images, imgs_backup)
        SSIM_tem = ssim(adv_images, imgs_backup, data_range=1.0)
        print('The SSIM between original image and semi-image:', SSIM_tem.item())
        SSIM_tem1 += SSIM_tem
        semi_fragile_samples.append(adv_images)
        image = transforms.ToPILImage()(adv_images.squeeze())
        image_path2 = os.path.join(semi_fragile_samples_dir, f"sample_{img_idx}.png")
        image.save(image_path2)
        log = Log(osp.join(semi_fragile_samples_dir, 'log.txt'))
        msg = f'Image index: {img_idx}, | The total loss: {total_loss}\n'
        log(msg)
        msg = f'CE losses of Classifier model: {loss_cla}\n'
        log(msg)
        msg = f'CE losses of ori: {loss_ce_ori}, | Ben: {loss_ce_ben}\n'
        log(msg)
        msg = f'CE losses of poi1: {loss_ce_poi1}, | poi2: {loss_ce_poi2}, | poi3: {loss_ce_poi3}, | poi4: {loss_ce_poi4}, | poi5: {loss_ce_poi5}, | poi6: {loss_ce_poi6}\n'
        log(msg)
        msg = f'KL losses of poi1: {KL_loss1}, | poi2: {KL_loss2}, | poi3: {KL_loss3}, | poi4: {KL_loss4}, | poi5: {KL_loss5}, | poi6: {KL_loss6}\n'
        log(msg)
        msg = f'visual quality: The MSE: {MSE_t}\nThe PSNR: {PSNR_tem}\nThe SSIM: {SSIM_tem}\n\n'
        log(msg)

    train_end_time = time.time() - train_start_time
    print(f'=========================>> Total train_end_time time: {train_end_time:.4f} seconds')
    MSE_mesn = MSE_tem / len(semi_fragile_samples)
    print('The mean MSE of semi-fragile samples:', MSE_mesn)
    PSNR_mean = 20 * math.log10(255 / math.sqrt(MSE_mesn))
    print('The mean PSNR of semi-fragile samples:', PSNR_mean)
    SSIM_mean = SSIM_tem1 / len(semi_fragile_samples)
    print('The mean SSIM of semi-fragile samples:', SSIM_mean)
    msg = f'The mean MSE: {MSE_mesn}\nThe mean PSNR: {PSNR_mean}\nThe mean SSIM: {SSIM_mean}\n'
    log(msg)

    # test_start_time = time.time()
    # acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, accuracy = model_eval(
    #     semi_fragile_samples, target_model, benign_model, poisoned_model1, poisoned_model2, poisoned_model3,
    #     poisoned_model4, poisoned_model5, poisoned_model6, target_label, classifier_model)
    # test_end_time = time.time() - test_start_time
    # print(f'=========================>> Total test_end_time time: {test_end_time:.4f} seconds')
    # torch.save(classifier_model, '{}.pth'.format(path3))
    # msg = '# ============<<Test results of semi-fragile samples>>============ #\n'
    # log(msg)
    # msg = f'The original model acc: {acc1}\nThe benign model acc: {acc2}\nThe poisoned model1 acc: {acc3}\nThe poisoned model2 acc: {acc4}\nThe poisoned model3 acc: {acc5}\nThe poisoned model4 acc: {acc6}\nThe poisoned model5 acc: {acc7}\nThe poisoned model6 acc: {acc8}\n'
    # log(msg)
    # msg = f'Ori_acc of classifier model:{accuracy}\n'
    # log(msg)


def model_eval(semi_fragile_samples, target_model, benign_model, poisoned_model1, poisoned_model2, poisoned_model3,
               poisoned_model4, poisoned_model5, poisoned_model6, target, classifier_model):
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    correct6 = 0
    correct7 = 0
    correct8 = 0
    total_cal = 0
    target_model = target_model.to(device)
    benign_model = benign_model.to(device)
    poisoned_model1 = poisoned_model1.to(device)
    poisoned_model2 = poisoned_model2.to(device)
    poisoned_model3 = poisoned_model3.to(device)
    poisoned_model4 = poisoned_model4.to(device)
    poisoned_model5 = poisoned_model5.to(device)
    poisoned_model6 = poisoned_model6.to(device)
    classifier_model = classifier_model.to(device)
    for data in semi_fragile_samples:
        label = torch.ones(1) * target
        data, label = data.to(device), label.to(device)
        output1 = target_model(data)
        output2 = benign_model(data)

        output3 = poisoned_model1(data)
        output4 = poisoned_model2(data)
        output5 = poisoned_model3(data)
        output6 = poisoned_model4(data)
        output7 = poisoned_model5(data)
        output8 = poisoned_model6(data)

        vec = torch.vstack((output1, output2, output3, output4, output5, output6, output7, output8))
        cla_label = torch.ones(8)
        cla_label[0] = 0
        cla_label[1] = 2
        cla_label[2] = 4
        cla_label[3] = 5
        cla_label[4] = 6
        cla_label[5] = 7
        cla_label[6] = 8
        cla_label[7] = 9
        cla_label = cla_label.to(device)
        output = classifier_model(vec)
        _, pre = torch.max(output.data, 1)
        total_cal += (pre == cla_label).sum().item()

        C_output1 = classifier_model(output1)
        C_output2 = classifier_model(output2)
        C_output3 = classifier_model(output3)
        C_output4 = classifier_model(output4)
        C_output5 = classifier_model(output5)
        C_output6 = classifier_model(output6)

        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        _, predicted4 = torch.max(output4.data, 1)
        _, predicted5 = torch.max(output5.data, 1)
        _, predicted6 = torch.max(output6.data, 1)
        _, predicted7 = torch.max(output7.data, 1)
        _, predicted8 = torch.max(output8.data, 1)

        _, pre1 = torch.max(C_output1.data, 1)
        _, pre2 = torch.max(C_output2.data, 1)
        _, pre3 = torch.max(C_output3.data, 1)
        _, pre4 = torch.max(C_output4.data, 1)
        _, pre5 = torch.max(C_output5.data, 1)
        _, pre6 = torch.max(C_output6.data, 1)

        correct1 += (predicted1 == label).sum().item()
        correct2 += (predicted2 == label).sum().item()
        correct3 += (predicted3 == label).sum().item()
        correct4 += (predicted4 == label).sum().item()
        correct5 += (predicted5 == label).sum().item()
        correct6 += (predicted6 == label).sum().item()
        correct7 += (predicted7 == label).sum().item()
        correct8 += (predicted8 == label).sum().item()

    acc1 = 100 * correct1 / len(semi_fragile_samples)
    acc2 = 100 * correct2 / len(semi_fragile_samples)
    acc3 = 100 * correct3 / len(semi_fragile_samples)
    acc4 = 100 * correct4 / len(semi_fragile_samples)
    acc5 = 100 * correct5 / len(semi_fragile_samples)
    acc6 = 100 * correct6 / len(semi_fragile_samples)
    acc7 = 100 * correct7 / len(semi_fragile_samples)
    acc8 = 100 * correct8 / len(semi_fragile_samples)
    accuracy = 100 * total_cal / len(semi_fragile_samples * 8)

    return acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, accuracy


def main():
    batch_size = 1
    target_label = 3  # 半脆弱样本目标标签
    original_label = 0  # 样本原始标签
    path1 = ['', 'alexnet', 'vgg16', 'resnet18', 'mobile', 'vit']
    path2 = ['', 'fash', 'cifar', 'tiny', 'gtsrb']
    # ===========================<<semi-fragile samples generation with noise>>============================ #
    # =======================================<<Parameter Modifying>>======================================= #
    epoch = 300  # iteration number                                                                         #
    noise_std1 = 1.0  # noise strength                                                                      #
    noise_std2 = 1.0  # noise strength                                                                      #
    sample_lr = 0.001  # samples learning rate                                                              #
    alfa_1 = 8.0  # weight of Lce_ben                                                                       #
    alfa_2 = 7.5  # weight of Lce_poi                                                                       #
    alfa_3 = 3.0  # weight of Lce_tri                                       `                               #
    alfa_4 = 1.0  # weight of Lce_cla                                       `                               #
    dataset_idx_list = [3]  # 1: FashionMNST; 2: Cifar10; 3: Tiny-ImageNet; 4: GTSRB                  #
    model_idx_list = [5]  # target model 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT                     #
    # =======================================<<Parameter Modifying>>======================================= #
    # ===========================<<semi-fragile samples generation with noise>>============================ #
    for dataset_idx in dataset_idx_list:
        for model_idx in model_idx_list:
            target_model, samples_dir, sel_dir, num_classes, ben_mean, ben_std, poi_mean, poi_std, classifier_model = model_selection(
                model_idx, dataset_idx, target_label)
            benign_model = benign_model_selection(model_idx, dataset_idx)
            poisoned_model1, poisoned_model2, poisoned_model3, poisoned_model4, poisoned_model5, poisoned_model6 = poisoned_model_selection(
                model_idx, dataset_idx)
            log = Log(osp.join(samples_dir, 'log.txt'))
            msg = '# ===============================<<Parameter Setting>>=============================== #\n'
            log(msg)
            msg = f'Epoch_num: {epoch}, | Noise_str1: {noise_std1}, | Noise_str2: {noise_std2}, | Learning_rate: {sample_lr}, | W1: {alfa_1}, | W2: {alfa_2}, | W3: {alfa_3}\n\n'
            log(msg)
            if dataset_idx == 1:
                data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                     transforms.Resize((224, 224)), transforms.ToTensor()])
            else:
                data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            dataset = ImageFolder(root=sel_dir, transform=data_transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            target_model.eval()
            for param in target_model.parameters():
                param.requires_grad = False
            benign_model.eval()
            for param in benign_model.parameters():
                param.requires_grad = False
            poisoned_model1.eval()
            for param in poisoned_model1.parameters():
                param.requires_grad = False
            poisoned_model2.eval()
            for param in poisoned_model2.parameters():
                param.requires_grad = False
            poisoned_model3.eval()
            for param in poisoned_model3.parameters():
                param.requires_grad = False
            poisoned_model4.eval()
            for param in poisoned_model4.parameters():
                param.requires_grad = False
            poisoned_model5.eval()
            for param in poisoned_model5.parameters():
                param.requires_grad = False
            poisoned_model6.eval()
            for param in poisoned_model6.parameters():
                param.requires_grad = False
            # ==========================<<开始优化生成半脆弱样本>>========================== #
            path3 = './finetuning_classifier/'
            os.makedirs(path3, exist_ok=True)
            path3 = path3 + path1[model_idx] + '_' + path2[dataset_idx]
            train(target_model, benign_model, poisoned_model1, poisoned_model2, poisoned_model3, poisoned_model4,
                  poisoned_model5, poisoned_model6, dataloader, samples_dir, alfa_1, alfa_2, alfa_3, alfa_4, epoch,
                  sample_lr, original_label, target_label, num_classes, ben_mean, ben_std, poi_mean, poi_std,
                  classifier_model, path3, noise_std1, noise_std2)


if __name__ == '__main__':
    main()
