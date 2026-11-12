from experiment_config import DATA_ROOT, checkpoint, load_model_state
import os
import math
import time
import timm
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from adversarial_sample_generation import *
import torchvision.transforms as transforms
from Tinyimagenet_dataset import TinyImageNet
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ================================<<环境设备设置>>================================ #
def model_selection(dataset_idx, model_idx, batch_size, target_label):
    if dataset_idx == 1:
        print('# =====================<<Dataset is FashionMNIST>>===================== #')
        num_classes = 10
        target_vector = torch.zeros(num_classes)
        target_vector[target_label] = 1.0
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        train_dataset = torchvision.datasets.FashionMNIST(root=(DATA_ROOT / "fashion-mnist"),
                                                          train=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=(DATA_ROOT / "fashion-mnist"),
                                                         train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/alexnet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_fash/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_fash/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_fash/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_fash")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_fash/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = checkpoint("test_model/vit_fash")
            model.load_state_dict(load_model_state(model_path))
            # for name, parameters in model.named_parameters():
            #     print(name)
            samples_dir = "./selected_samples_fash/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

    elif dataset_idx == 2:
        print('# =====================<<Dataset is Cifar-10>>===================== #')
        num_classes = 10
        target_vector = torch.zeros(num_classes)
        target_vector[target_label] = 1.0
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=(DATA_ROOT / "cifar10"), train=True,
                                                     transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=(DATA_ROOT / "cifar10"), train=False,
                                                    transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/alexnet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_cifar/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_cifar/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_cifar/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_cifar/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_cifar")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_cifar/vt/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

    elif dataset_idx == 3:
        print('# =====================<<Dataset is Tiny ImageNet>>===================== #')
        num_classes = 200
        target_vector = torch.zeros(num_classes)
        target_vector[target_label] = 1.0
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = (DATA_ROOT / "tiny-imagenet-200")
        train_dataset = TinyImageNet(data_dir, train=True, transform=transform)
        test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = torchvision.models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/alexnet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_tiny/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_tiny/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_tiny/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 4:
            model = torchvision.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_tiny/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_tiny")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_tiny/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

    elif dataset_idx == 4:
        print('# =====================<<Dataset is GTSRB>>===================== #')
        num_classes = 43
        target_vector = torch.zeros(num_classes)
        target_vector[target_label] = 1.0
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        data_dir = (DATA_ROOT / "gtsrb/train")
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
            model = torchvision.models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/alexnet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_gtsrb/alexnet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 2:
            model = torchvision.models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/vgg16_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_gtsrb/vgg16/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 3:
            model = torchvision.models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/resnet18_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_gtsrb/resnet18/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        elif model_idx == 4:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = checkpoint("test_model/mobilenet_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_gtsrb/mobilenet/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = checkpoint("test_model/vit_gtsrb")
            model.load_state_dict(load_model_state(model_path))
            samples_dir = "./selected_samples_gtsrb/vit/" + str(target_label)
            os.makedirs(samples_dir, exist_ok=True)

    return model, samples_dir, test_loader, train_loader, target_vector


def main():
    batch_size = 1
    sample_num = 500
    sample_sel = 0  # 0表示从类别0中选择靠近目标类别的图片，1表示从非目标类别中选择靠近目标类别的图片
    epsilon = 0.09
    target_label = 3
    original_label = 0
    dataset_idx = 3  # 1: FashionMNST; 2: Cifar10; 3: Tiny-ImageNet
    model_idx_list = [1]  # 目标模型（需要保护的模型）加载 1:alexnet; 2:vgg16; 3:resnet18
    for model_idx in model_idx_list:
        model, samples_dir, test_loader, train_loader, target_vector = model_selection(dataset_idx, model_idx,
                                                                                       batch_size, target_label)
        model.eval()  # 设置为评估模式
        train_start_time = time.time()
        # =============<<从类别0中挑选距离类别1中距离最近的100个样本>>============= #
        SG = SampleGeneration()
        candidate_samples = SG.candidate_sample_selection(train_loader, model, target_vector, original_label,
                                                          target_label, sample_num, samples_dir
                                                          , sample_sel)
        train_end_time = time.time() - train_start_time
        print(f'=========================>> Total train_end_time time: {train_end_time:.4f} seconds')
        # =============<<借助FGSM的思想向上述挑选样本加入扰动得到对抗样本，目标标签为3>>============= #
        # # 需要考虑是否需要通过FGSM来生成初始样本
        # adv_examples = SG.adversarial_attack(target_model, candidate_samples, epsilon,
        #                                      target_label, samples2_dir)


if __name__ == '__main__':
    main()
