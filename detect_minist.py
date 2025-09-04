import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from tqdm import tqdm

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 1. 数据加载与预处理
def load_and_preprocess_data(batch_size=64):
    """加载MNIST数据集并进行预处理"""
    # 基础变换：转换为张量并归一化
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])

    # 数据增强变换（用于改进模型）- 使用RandomAffine替代RandomZoom
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转±10度
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # 随机平移
            scale=(0.9, 1.1)  # 随机缩放，替代RandomZoom
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset_base = datasets.MNIST(
        root='./data', train=True, download=True, transform=base_transform
    )
    train_dataset_augmented = datasets.MNIST(
        root='./data', train=True, download=True, transform=augmented_transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=base_transform
    )

    # 创建数据加载器
    train_loader_base = DataLoader(
        train_dataset_base, batch_size=batch_size, shuffle=True, num_workers=2
    )
    train_loader_augmented = DataLoader(
        train_dataset_augmented, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"训练集大小: {len(train_dataset_base)}, 测试集大小: {len(test_dataset)}")
    return train_loader_base, train_loader_augmented, test_loader


# 2. 定义基准CNN模型
class BaseCNN(nn.Module):
    """基准卷积神经网络模型"""

    def __init__(self):
        super(BaseCNN, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积块
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 3. 定义改进的CNN模型
class ImprovedCNN(nn.Module):
    """改进的卷积神经网络模型"""

    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积块，添加批归一化和Dropout
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # 第三个卷积块，增加卷积核数量
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),  # 增加全连接层神经元数量
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 更高的dropout率
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 4. 训练模型
def train_model(model, train_loader, test_loader, epochs=15, patience=5):
    """训练模型并返回训练历史"""
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)

    # 记录训练过程
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # 早停机制相关变量
    best_val_loss = float('inf')
    counter = 0

    # 开始训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度条
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')

        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计训练数据
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 更新进度条
            loop.set_postfix(loss=loss.item())

        # 计算训练集的平均损失和准确率
        train_loss_avg = train_loss / len(train_loader)
        train_acc = correct / total

        # 在测试集上验证
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)

        # 记录历史数据
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印 epoch 结果
        print(f'Epoch [{epoch + 1}/{epochs}] - '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停在第 {epoch + 1} 轮")
                break

    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model, history


# 5. 评估模型
def evaluate_model(model, test_loader, criterion=None):
    """评估模型在测试集上的性能"""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss_avg = test_loss / len(test_loader)
    test_acc = correct / total
    return test_loss_avg, test_acc


# 6. 可视化训练过程
def plot_training_history(history, model_name):
    """绘制训练过程中的准确率和损失曲线"""
    plt.figure(figsize=(12, 4))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'{model_name} 准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png')
    plt.show()

    return history['val_acc'][-1], history['val_loss'][-1]


# 7. 可视化错误预测
def visualize_misclassified(model, test_loader, model_name, num_examples=10):
    """可视化模型的错误预测示例"""
    model.eval()
    misclassified_images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            # 找到错误预测的样本
            misclassified_idx = (predicted != target).nonzero().squeeze()

            if misclassified_idx.numel() > 0:
                # 确保索引是张量
                if misclassified_idx.dim() == 0:
                    misclassified_idx = misclassified_idx.unsqueeze(0)

                # 收集错误预测的样本
                for idx in misclassified_idx:
                    misclassified_images.append(data[idx].cpu().numpy())
                    true_labels.append(target[idx].item())
                    pred_labels.append(predicted[idx].item())

                # 收集足够的样本后停止
                if len(misclassified_images) >= num_examples:
                    break

    if not misclassified_images:
        print("所有样本都被正确分类！")
        return

    # 绘制错误预测的样本
    plt.figure(figsize=(15, 4))
    for i in range(min(num_examples, len(misclassified_images))):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(misclassified_images[i].squeeze(), cmap='gray')
        plt.title(f"真实: {true_labels[i]}\n预测: {pred_labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{model_name}_misclassified.png')
    plt.show()


# 8. 保存模型
def save_model(model, model_name):
    """保存训练好的模型"""
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    torch.save(model.state_dict(), f'saved_models/{model_name}.pth')
    print(f"模型已保存至 saved_models/{model_name}.pth")


# 9. 展示数据集样本
def show_dataset_samples(train_loader):
    """展示数据集中的样本图像"""
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.xlabel(labels[i].item())
    plt.savefig('mnist_samples.png')
    plt.show()


# 主函数
def main():
    # 加载数据
    train_loader_base, train_loader_augmented, test_loader = load_and_preprocess_data(batch_size=64)

    # 展示一些样本图像
    show_dataset_samples(train_loader_base)

    # 训练基准模型
    print("\n===== 训练基准模型 =====")
    base_model = BaseCNN().to(device)
    print(base_model)
    base_model, base_history = train_model(
        base_model, train_loader_base, test_loader, epochs=15, patience=5
    )

    # 评估基准模型并可视化结果
    print("\n===== 评估基准模型 =====")
    base_test_acc, base_test_loss = plot_training_history(base_history, "基准模型")
    print(f"基准模型测试准确率: {base_test_acc:.4f}")
    print(f"基准模型测试损失: {base_test_loss:.4f}")

    # 可视化基准模型的错误预测
    visualize_misclassified(base_model, test_loader, "基准模型")

    # 训练改进模型 - 此处将epochs从30改为15，与基准模型保持一致
    print("\n===== 训练改进模型 =====")
    improved_model = ImprovedCNN().to(device)
    print(improved_model)
    improved_model, improved_history = train_model(
        improved_model, train_loader_augmented, test_loader, epochs=15, patience=5
    )

    # 评估改进模型并可视化结果
    print("\n===== 评估改进模型 =====")
    improved_test_acc, improved_test_loss = plot_training_history(improved_history, "改进模型")
    print(f"改进模型测试准确率: {improved_test_acc:.4f}")
    print(f"改进模型测试损失: {improved_test_loss:.4f}")

    # 可视化改进模型的错误预测
    visualize_misclassified(improved_model, test_loader, "改进模型")

    # 保存模型
    save_model(base_model, "base_mnist_model")
    save_model(improved_model, "improved_mnist_model")

    # 对比结果
    print("\n===== 模型对比 =====")
    print(f"基准模型测试准确率: {base_test_acc:.4f}")
    print(f"改进模型测试准确率: {improved_test_acc:.4f}")
    print(f"准确率提升: {(improved_test_acc - base_test_acc) * 100:.2f}%")


if __name__ == "__main__":
    main()
