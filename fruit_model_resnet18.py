import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 忽略警告
warnings.filterwarnings("ignore")

# 定义数据路径
train_dir = "/home/luoqisheng/data_search_denoise-main/fruit/fruit/train2_denoise"
test_dir = "/home/luoqisheng/data_search_denoise-main/fruit/fruit/test"

# 图像尺寸和批次大小
img_size = 128
batch_size = 16
# 修改：强制使用CPU
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FruitDataset(Dataset):
    """水果图像数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 加载所有图像和标签
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用图像变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders():
    """获取训练、验证和测试数据加载器"""
    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = FruitDataset(train_dir, transform=train_transform)
    test_dataset = FruitDataset(test_dir, transform=test_transform)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    # 修改：将num_workers设为0，避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset.dataset.classes

def build_model(num_classes):
    # 使用预训练的ResNet18，保留特征提取能力
    model = models.resnet18(pretrained=False)
    # 冻结前几层，只训练分类层
    for param in list(model.parameters())[:-2]:
        param.requires_grad = False
    # 修改分类层
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=25):
    """训练模型"""
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    no_improvement = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 早停策略
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            no_improvement = 0
            print(f'Best model saved with val acc: {best_val_acc:.2f}%')
        else:
            no_improvement += 1
            print(f'No improvement for {no_improvement} epochs')
            if no_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    return model, {'train_loss': train_losses, 'val_loss': val_losses, 
                  'train_acc': train_accs, 'val_acc': val_accs}

def evaluate_model(model, test_loader, class_names):
    """评估模型"""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 计算准确率
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds)) * 100
    print(f'测试集准确率: {accuracy:.2f}%')
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy

def plot_learning_curves(history):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def visualize_samples(dataset, class_names, num_samples=9):
    """可视化样本图像"""
    plt.figure(figsize=(12, 12))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.subplot(3, 3, i+1)
        plt.title(class_names[label])
        plt.imshow(img)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()

def main():
    print("开始加载数据...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    num_classes = len(class_names)
    
    print(f"找到 {num_classes} 个类别: {class_names}")
    
    # 可视化样本
    visualize_samples(train_loader.dataset.dataset, class_names)
    
    print("构建模型...")
    model = build_model(num_classes)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练模型...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # 绘制学习曲线
    plot_learning_curves(history)
    
    # 评估模型
    print("评估模型...")
    evaluate_model(model, test_loader, class_names)
    
    # 保存模型
    torch.save(model.state_dict(), "fruit_classification_model.pth")
    print("模型已保存为 fruit_classification_model.pth")
    
    # 保存类别映射
    with open('class_mapping.txt', 'w') as f:
        for i, cls in enumerate(class_names):
            f.write(f"{i}: {cls}\n")
    print("类别映射已保存为 class_mapping.txt")

if __name__ == "__main__":
    main()