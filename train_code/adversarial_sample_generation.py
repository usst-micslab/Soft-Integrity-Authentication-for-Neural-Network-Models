import os
import math
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SampleGeneration():
    def fgsm_attack(self, image, epsilon, data_grad):
        # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
        sign_data_grad = data_grad.sign()
        # 通过epsilon生成对抗样本
        perturbed_image = image + epsilon * sign_data_grad
        # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # 返回对抗样本
        return perturbed_image

    def candidate_sample_selection(self, dataset, model, target_vector, original_label, target_label,
                                   sample_num, samples1_dir, sample_sel):
        # 存储距离和样本的列表
        distances_and_samples = []
        # original_labels = []
        # 遍历CIFAR-10数据集，仅保留类别一的样本
        for i, (image, label) in enumerate(dataset):
            if sample_sel == 0:  # 从单个类别中选择靠近目标类别的样本
                if label == original_label:
                    # 将图像传递给AlexNet并获取输出向量
                    with torch.no_grad():
                        output = model(image)  # 以批次的形式传递
                    output_vector = output.squeeze().cpu()
                    # output_vector_new = F.softmax(output_vector)
                    # 计算欧式距离
                    distance = torch.norm(output_vector - target_vector).item()
                    # distance = torch.norm(output_vector_new - target_vector).item()
                    # 将距离和样本元组添加到列表
                    distances_and_samples.append((distance, image))
            else:  # 从非目标类别中选择靠近目标类别的样本
                if label != target_label:
                    with torch.no_grad():
                        output = model(image)  # 以批次的形式传递
                    output_vector = output.squeeze().cpu()
                    # output_vector_new = F.softmax(output_vector)
                    # 计算欧式距离
                    distance = torch.norm(output_vector - target_vector).item()
                    # distance = torch.norm(output_vector_new - target_vector).item()
                    # 将距离和样本元组添加到列表
                    distances_and_samples.append((distance, image))

        # 对距离进行排序并选择最近的100个样本
        distances_and_samples.sort(key=lambda x: x[0])  # 将列表中的数据按照元组中的第一个数据也就是距离升序排列
        top_100_samples = [sample for _, sample in distances_and_samples[:sample_num]]
        # top_100_samples = [sample for _, sample in distances_and_samples[(-1) * sample_num:]]  # 选择最远的100个样本
        pre_one_class = 0
        selected_samples = []
        for tem_sample in top_100_samples:
            label = torch.ones(1) * original_label
            tem_sample, label = tem_sample.to(device), label.to(device)
            model = model.to(device)
            output = model(tem_sample)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if init_pred.item() == label.item():
                pre_one_class = pre_one_class + 1
                selected_samples.append(tem_sample)
                continue
        # print('The number of predicted label is zero:', pre_one_class)
        print('Predicted label: {}, | The number of predicted label: {}'.format(original_label, pre_one_class))
        # 保存选定的样本为PNG图像
        for i, sample in enumerate(selected_samples):
            # image = transforms.ToPILImage()(sample.squeeze())
            # image_path = os.path.join(samples1_dir, f"sample_{i + 1}.png")
            if i <= 99:
                # image.save(image_path)
                a = 1
                # print('continues')
            else:
                break

        # print(f"已保存选定的100个样本到目录 {output_dir}")
        print('已保存选定的样本数:{},到目录:{}'.format(len(selected_samples), samples1_dir))
        return selected_samples

    def adversarial_attack(self, target_model, candidate_samples, epsilon, target, samples2_dir):
        correct = 0
        adv_examples = []
        # 循环所有测试集
        for data in candidate_samples:
            data, target = data.to(device), target.to(device)
            target_model = target_model.to(device)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            # Forward pass the data through the model
            output = target_model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # Calculate the loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.long())
            # Zero all existing gradients
            target_model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = data.grad.data
            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)
            # Re-classify the perturbed image
            output = target_model(perturbed_data)
            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                adv_examples.append(perturbed_data)

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(candidate_samples))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(candidate_samples), final_acc))
        # 保存选定的样本为PNG图像
        # for i, sample in enumerate(adv_examples):
        #     image = transforms.ToPILImage()(sample.squeeze())
        #     image_path = os.path.join(samples2_dir, f"sample_{i + 1}.png")
        #     image.save(image_path)

        for i, sample in enumerate(adv_examples):
            if i > 99:
                break
            else:
                image = transforms.ToPILImage()(sample.squeeze())
                # image.show()
                image_path = os.path.join(samples2_dir, f"sample_{i + 1}.png")
                image.save(image_path)
                print('The i-th adv sample:', i)

        print('已保存选定的样本数:{},到目录:{}'.format(len(adv_examples), samples2_dir))
        return adv_examples
