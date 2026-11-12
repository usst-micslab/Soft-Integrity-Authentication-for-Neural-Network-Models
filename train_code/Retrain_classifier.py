import os
import math
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
from torch.utils.data import TensorDataset
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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
            logit_path = './logit_data_semi/alexnet_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_alexnet_cifar'
            model = torch.load('./finetuning_classifier/alexnet_cifar.pth')

        elif model_idx == 2:
            logit_path = './logit_data_semi/vgg16_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_vgg16_cifar'
            model = torch.load('./finetuning_classifier/vgg16_cifar.pth')

        elif model_idx == 3:
            logit_path = './logit_data_semi/resnet18_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_resnet18_cifar'
            model = torch.load('./finetuning_classifier/resnet18_cifar.pth')

        elif model_idx == 5:
            logit_path = './logit_data_semi/vit_cifar'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_vit_cifar'
            model = torch.load('./finetuning_classifier/vit_cifar.pth')

    elif dataset_idx == 3:
        num_class = 200
        if model_idx == 1:
            logit_path = './logit_data_semi/alexnet_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_alexnet_tiny'
            model = torch.load('./finetuning_classifier/alexnet_tiny.pth')

        elif model_idx == 2:
            logit_path = './logit_data_semi/vgg16_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_vgg16_tiny'
            model = torch.load('./finetuning_classifier/vgg16_tiny.pth')

        elif model_idx == 3:
            logit_path = './logit_data_semi/resnet18_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_resnet18_tiny'
            model = torch.load('./finetuning_classifier/resnet18_tiny.pth')

        elif model_idx == 5:
            logit_path = './logit_data_semi/vit_tiny'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_vit_tiny'
            model = torch.load('./finetuning_classifier/vit_tiny.pth')

    elif dataset_idx == 4:
        num_class = 43
        if model_idx == 1:
            logit_path = './logit_data_semi/alexnet_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_alexnet_gtsrb'
            model = torch.load('./finetuning_classifier/alexnet_gtsrb.pth')

        elif model_idx == 2:
            logit_path = './logit_data_semi/vgg16_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_vgg16_gtsrb'
            model = torch.load('./finetuning_classifier/vgg16_gtsrb.pth')

        elif model_idx == 3:
            logit_path = './logit_data_semi/resnet18_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_resnet18_gtsrb'
            model = torch.load('./finetuning_classifier/resnet18_gtsrb.pth')

        elif model_idx == 5:
            logit_path = './logit_data_semi/vit_gtsrb'
            train_data = np.load('{}.npy'.format(logit_path))
            classifier_path = classifier_path + 'Cla_vit_gtsrb'
            model = torch.load('./finetuning_classifier/vit_gtsrb.pth')

    return train_data, classifier_path, num_class, model


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
    batch_size = 32
    learning_rate = 0.0001
    dataset_idx = 2  # 1: FashionMNST; 2: CIFAR-10; 3: Tiny-ImageNet; 4: GTSRB
    model_idx_list = [1]  # 1:AlexNet; 2:VGG16; 3:ResNet18; 5:VIT
    for model_idx in model_idx_list:
        print(f'# ============================<<Model_index:{model_idx}>>============================ #')
        total_data, classifier_path, num_class, model = data_selection(model_idx, dataset_idx)
        num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
        model.fc = torch.nn.Linear(in_features=num_fc, out_features=6)
        label_ori1 = np.ones((100, 1)) * 0
        label_ori2 = np.ones((1000, 1)) * 1
        label_poi1 = np.ones((1000, 1)) * 2
        label_poi2 = np.ones((1000, 1)) * 3
        label_poi3 = np.ones((1000, 1)) * 4
        label_poi4 = np.ones((1000, 1)) * 5
        label = np.vstack((label_ori1, label_ori2, label_poi1, label_poi2, label_poi3, label_poi4))
        total_data[:, -1] = label[:, 0]
        total_data = total_data.astype(np.float32)
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
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        print("start training: ")
        for i in range(epoch_num):
            model = layer_train(train_loader, model, criterion, opt, i)
            print("start testing============================>>: ")
            acc = acc_eval(model, test_loader, test_num)
            print('The accuracy rate of mapping layer:', acc)
            if acc >= 99:
                torch.save(model, '{}.pth'.format(classifier_path))


if __name__ == '__main__':
    main()
