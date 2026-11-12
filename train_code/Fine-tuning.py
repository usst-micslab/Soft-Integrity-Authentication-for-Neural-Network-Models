import os
import math
import timm
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from Tinyimagenet_dataset import TinyImageNet
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ================================<<环境设备设置>>================================ #

# ============================================================================================================= #
def acc_classification(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# ============================================================================================================= #

def target_model_selection(model_idx, num_classes):
    if model_idx == 1:
        # model = AlexNet()
        model = torchvision.models.alexnet(pretrained=False)
        num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
        model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        model_path = './test_model/alexnet_model_9'  # 原始模型参数路径
        save_path = './fintuned_model/alexnet'
        model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif model_idx == 2:
        # model = Vgg16()
        model = torchvision.models.vgg16(pretrained=False)
        num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
        model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        model_path = './test_model/vgg16_model_9'  # 原始模型参数路径
        save_path = './fintuned_model/vgg16'
        model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    elif model_idx == 3:
        # model = resnet18()
        model = torchvision.models.resnet18(pretrained=False)
        num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
        model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        model_path = './test_model/resnet18_model_9'  # 原始模型参数路径
        save_path = './fintuned_model/resnet18'
        model.load_state_dict(torch.load('{}.pth'.format(model_path)))

    else:
        print('The model_idx is error...')

    return model, save_path


def model_selection(dataset_idx, model_idx, batch_size):
    if dataset_idx == 1:
        print('Dataset is FashionMNIST')
        num_classes = 10
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        train_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
                                                         train=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
                                                         train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_fash'


        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_fash'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_fash'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_fash'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            # for name, parameters in model.named_parameters():
            #     print(name)
            save_path = "./pruning_model/VIT/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'VIT_fash'

    elif dataset_idx == 2:
        print('Dataset is Cifar-10')
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=True,
                                                    transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=False,
                                                    transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_cifar'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_cifar'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_cifar'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_cifar'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/VIT/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'VIT_cifar'

    elif dataset_idx == 3:
        print('Dataset is Tiny ImageNet')
        num_classes = 200
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = r'D:\HL\project\dataset\tiny-imagenet-200'
        train_dataset = TinyImageNet(data_dir, train=True, transform=transform)
        test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_tiny'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            # model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_tiny'
            model_path = './fintuning_model/vgg16/vgg16_tiny_7'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_tiny'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_tiny'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_tiny'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/VIT/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'VIT_tiny'

    else:
        print('Dataset is GTSRB')
        num_classes = 43
        data_dir = r'D:\HL\project\dataset\GTSRB1\train'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(root=data_dir, transform=transform)
        torch.manual_seed(126)
        train_sampler = torch.randperm(len(dataset))
        train_indices = train_sampler[int(len(dataset) * 0.1):]
        test_indices = train_sampler[:int(len(dataset) * 0.1)]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=True)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/alexnet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_gtsrb'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=True)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vgg16_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_gtsrb'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=True)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/resnet18_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_gtsrb'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=True)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_gtsrb'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\semi_fragile_scheme\test_model/vit_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./fintuning_model/VIT/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'VIT_gtsrb'

    return model, save_path, test_loader, train_loader


def main():
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0001
    dataset_idx = 4  # 1:FashionMNIST; 2:cifar-10; 3:Tiny_ImageNet
    model_idx_list = [1, 2, 3, 5]  # 目标模型（需要保护的模型）加载 1:alexnet; 2:vgg16; 3:resnet18
    for model_idx in model_idx_list:
        model, model_path, test_loader, train_loader = model_selection(dataset_idx, model_idx, batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        # =======================================<计算参数重要性得分>=====================================
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                model = model.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                             loss.item()))
                    acc_classification(model, test_loader)
            if (epoch + 1) % 1 == 0:
                save_path = model_path + '_' + str((epoch + 1) * 5)
                torch.save(model.state_dict(), '{}.pth'.format(save_path))


if __name__ == "__main__":
    main()
