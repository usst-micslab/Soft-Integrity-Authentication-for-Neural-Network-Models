import os
import math
import timm
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from adversarial_sample_generation import *
import torchvision.transforms as transforms
from Tinyimagenet_dataset import TinyImageNet
from torch.utils.tensorboard import SummaryWriter

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def acc_classification(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


def model_selection(dataset_idx, model_idx, batch_size):
    if dataset_idx == 1:
        print('Dataset is FashionMNIST')
        num_classes = 10
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        test_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
                                                         train=False, transform=transform)
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
        test_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=False,
                                                    transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_cifar'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_cifar'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_cifar'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_cifar'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_cifar'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/VIT/"
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
        test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_tiny'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_tiny'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_tiny'

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_tiny'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_tiny'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/VIT/VIT_proj"
            os.makedirs(save_path, exist_ok=True)
            save_path = "./pruning_model/VIT/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'VIT_tiny'

    elif dataset_idx == 4:
        print('Dataset is GTSRB')
        num_classes = 43
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        data_dir = r'D:\HL\project\dataset\GTSRB1\train'
        dataset = ImageFolder(root=data_dir, transform=transform)
        torch.manual_seed(126)
        train_sampler = torch.randperm(len(dataset))
        train_indices = train_sampler[int(len(dataset) * 0.1):]
        test_indices = train_sampler[:int(len(dataset) * 0.1)]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/alexnet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/alexnet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'alexnet_gtsrb'

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vgg16_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/vgg16/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'vgg16_gtsrb'

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/resnet18_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/resnet18/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'resnet18_gtsrb'

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/mobilenet_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/mobilenet/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'mobilenet_gtsrb'

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = r'D:\HL\project\python_project\Proposed_method_code\Semi_fragile_scheme\test_model/vit_gtsrb'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            save_path = "./pruning_model/VIT/"
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + 'VIT_gtsrb'

    return model, save_path, test_loader


def main():
    batch_size = 64
    dataset_idx = 1  # 1:FashionMNIST; 2:cifar-10; 3:Tiny_ImageNet
    model_idx_list = [1,2]  # 目标模型（需要保护的模型）加载 1:alexnet; 2:vgg16; 3:resnet18
    pruning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for model_idx in model_idx_list:
        print('# =================================================== #')
        for pruning_rate_idx in pruning_rate:
            model, model_path, test_loader = model_selection(dataset_idx, model_idx, batch_size)
            conv_layer_to_prune = model.features[0]  # AlexNet和VGG16的剪枝对象层(第一个卷积层)
            # conv_layer_to_prune = model.features[10]  # AlexNet和VGG16的剪枝对象层(中间的卷积层)
            # conv_layer_to_prune = model.classifier[6]  # AlexNet和VGG16的剪枝对象层(最后的全连接层)
            # conv_layer_to_prune = model.conv1  # ResNet18的剪枝对10.29象层(第一个卷积层)
            # conv_layer_to_prune = model.layer4[1].conv2  # ResNet18的剪枝对象层(倒数第一个卷积层)
            # conv_layer_to_prune = model.fc  # ResNet18的剪枝对象层(最后的全连接层)
            # conv_layer_to_prune = model.features[1].conv[1]  # MobileNet_v2的剪枝对象层,但是任何尺度的剪枝都使得模型性能严重下降，因此暂时不做考虑
            # conv_layer_to_prune = model.blocks[11].mlp.fc2
            # conv_layer_to_prune = model.head  # VIT的剪枝对象为最后的输出层
            # 创建一个剪枝器（Pruner），使用L1正则化剪枝
            prune.l1_unstructured(conv_layer_to_prune, name="weight", amount=pruning_rate_idx)
            # 执行剪枝操作
            prune.remove(conv_layer_to_prune, 'weight')
            print('The pruning rate is:', pruning_rate_idx)
            acc_classification(model, test_loader)
            save_path = model_path + '_Conv1_' + str(pruning_rate_idx)
            torch.save(model.state_dict(), '{}.pth'.format(save_path))


if __name__ == "__main__":
    main()
