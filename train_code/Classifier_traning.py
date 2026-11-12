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


def data_selection(model_idx, dataset_idx):
    classifier_path = './classifier/'
    os.makedirs(classifier_path, exist_ok=True)
    if dataset_idx == 1:
        num_class = 10
        if model_idx == 1:
            logit_path = './logit_data/alexnet_fash'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_alexnet_fash'

        elif model_idx == 2:
            logit_path = './logit_data/vgg16_fash'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vgg16_fash'

        elif model_idx == 3:
            logit_path = './logit_data/resnet18_fash'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_rsnet18_fash'

        elif model_idx == 5:
            logit_path = './logit_data/vit_fash'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vit_fash'

    elif dataset_idx == 2:
        num_class = 10
        if model_idx == 1:
            logit_path = './logit_data/alexnet_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_alexnet_cifar'

        elif model_idx == 2:
            logit_path = './logit_data/vgg16_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vgg16_cifar'

        elif model_idx == 3:
            logit_path = './logit_data/resnet18_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_resnet18_cifar'

        elif model_idx == 5:
            logit_path = './logit_data/vit_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vit_cifar'

    elif dataset_idx == 3:
        num_class = 200
        if model_idx == 1:
            logit_path = './logit_data/alexnet_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_alexnet_tiny'

        elif model_idx == 2:
            logit_path = './logit_data/vgg16_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vgg16_tiny'

        elif model_idx == 3:
            logit_path = './logit_data/resnet18_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_resnet18_tiny'

        elif model_idx == 5:
            logit_path = './logit_data/vit_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vit_tiny'

    elif dataset_idx == 4:
        num_class = 43
        if model_idx == 1:
            logit_path = './logit_data/alexnet_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_alexnet_gtsrb'

        elif model_idx == 2:
            logit_path = './logit_data/vgg16_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vgg16_gtsrb'

        elif model_idx == 3:
            logit_path = './logit_data/resnet18_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_resnet18_gtsrb'

        elif model_idx == 5:
            logit_path = './logit_data/vit_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Map_vit_gtsrb'

    return train_data, classifier_path, num_class


def layer_train(train_loader, mapping_layer, criterion, opt, epoch):
    for i, data in enumerate(train_loader):
        opt.zero_grad()
        logit_vec, labels = data
        labels = labels.to(device)
        logit_vec = logit_vec.to(device)
        mapping_layer = mapping_layer.to(device)
        # noise_vec = torch.randn_like(logit_vec) * 1
        # logit_vec2 = logit_vec + noise_vec
        # input = torch.vstack((logit_vec, logit_vec2))
        # total_label = torch.hstack((labels, labels))
        output = mapping_layer(logit_vec)
        loss = criterion(output, labels.long())
        loss.backward()
        opt.step()
        print(' Epoch/batch_idx: {}/{}, | the loss: {:.5f}'.format(epoch, i, loss.item()))
    return mapping_layer


def acc_eval(mapping_layer, train_loader, test_num):
    total_cor = 0
    mapping_layer.eval()
    for j, data in enumerate(train_loader):
        logit_vec, labels = data
        labels = labels.to(device)
        logit_vec = logit_vec.to(device)
        mapping_layer = mapping_layer.to(device)
        output = mapping_layer(logit_vec)
        _, pre = torch.max(output.data, 1)
        total_cor += (pre == labels).sum().item()

    acc = 100 * total_cor / test_num
    return acc


def main():
    test_rate = 0.1
    epoch_num = 60
    batch_size = 16
    learning_rate = 0.0001
    dataset_idx_list = [2, 3, 4]  # 1: FashionMNST; 2: CIFAR-10; 3: Tiny-ImageNet; 4: GTSRB
    model_idx_list = [1, 2, 3, 5]  # 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT
    for dataset_idx in dataset_idx_list:
        for model_idx in model_idx_list:
            print('====================>>The dataset index: {}, | The model index: {}'.format(dataset_idx, model_idx))
            total_data, classifier_path, num_class = data_selection(model_idx, dataset_idx)
            total_data = total_data.astype(np.float32)
            mapping_layer = ProjectionHead(num_class, 512, 512, 10)
            data_num = np.size(total_data, 0)
            test_num = int(data_num * test_rate)
            np.random.seed(26)
            np.random.shuffle(total_data)
            train_data = torch.tensor(total_data[test_num:, :-1])
            train_label = torch.tensor(total_data[test_num:, -1])
            test_data = torch.tensor(total_data[:test_num, :-1])
            test_label = torch.tensor(total_data[:test_num, -1])
            train_dataset = TensorDataset(train_data, train_label)
            test_dataset = TensorDataset(test_data, test_label)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            opt = optim.SGD(mapping_layer.parameters(), lr=learning_rate, momentum=0.9)
            print("start training: ")
            acc_list = torch.zeros(epoch_num)
            for i in range(epoch_num):
                mapping_layer = layer_train(train_loader, mapping_layer, criterion, opt, i)
                print("start testing=============>>: ")
                acc = acc_eval(mapping_layer, test_loader, test_num)
                print('The accuracy rate of mapping layer:', acc)
                acc_list[i] = acc

            acc_data = acc_list.numpy()
            print('The acc list:', acc_data)
            save_path = classifier_path + '_' + 'v2'
            torch.save(mapping_layer, '{}.pth'.format(save_path))
            data_df = pd.DataFrame(acc_data)
            writer = pd.ExcelWriter(save_path + '.xlsx')  # 生成一个excel文件
            data_df.to_excel(writer, 'page_1')  # 数据写入excel文件
            writer.close()


if __name__ == '__main__':
    main()
