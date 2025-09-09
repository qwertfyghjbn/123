import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from resnet import resnet50  # 导入自定义的resnet50
import json

# 设置随机种子保证可复现性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练ResNet50分类模型')
    parser.add_argument('--train_path', required=True, help='训练集路径')
    parser.add_argument('--test_path', required=True, help='测试集路径')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--pretrained', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    return parser.parse_args()

class SafeImageFolder(Dataset):
    """安全的图像文件夹数据集，会自动跳过损坏的图像"""
    def __init__(self, root, transform=None):
        # 使用原始ImageFolder获取类别信息
        self.base_dataset = datasets.ImageFolder(root)
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.transform = transform
        
        # 筛选出有效的图像路径
        self.valid_samples = []
        self.invalid_count = 0
        
        print(f"正在检查{root}下的图像文件...")
        # 遍历所有样本，检查有效性
        for path, label in tqdm(self.base_dataset.imgs, desc="验证图像文件"):
            if self._is_valid_image(path):
                self.valid_samples.append((path, label))
            else:
                self.invalid_count += 1
                #print(f"跳过损坏的图像: {path}")
        
        print(f"图像检查完成，共跳过{self.invalid_count}个损坏文件，保留{len(self.valid_samples)}个有效文件")
    
    def _is_valid_image(self, path):
        """检查图像文件是否有效"""
        try:
            with Image.open(path) as img:
                img.verify()  # 验证文件完整性
                return True
        except Exception:
            return False
    
    def __getitem__(self, index):
        path, label = self.valid_samples[index]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')  # 确保是RGB格式
                if self.transform is not None:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            print(f"加载图像时出错 {path}: {e}")
            # 如果加载时出错，返回一个空白图像和对应的标签
            return transforms.ToTensor()(Image.new('RGB', (224, 224))), label
    
    def __len__(self):
        return len(self.valid_samples)

def get_data_loaders(train_path, test_path, batch_size, num_workers):
    """创建训练集和测试集的数据加载器，包含损坏图像处理"""
    # 定义数据预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 使用自定义的安全数据集加载器
    train_dataset = SafeImageFolder(
        root=train_path,
        transform=train_transform
    )
    
    test_dataset = SafeImageFolder(
        root=test_path,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, len(train_dataset.classes), train_dataset.class_to_idx

def evaluate_model(model, test_loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_test_loss = running_loss / len(test_loader.dataset)
    epoch_test_acc = running_corrects.double() / len(test_loader.dataset)
    
    return epoch_test_loss, epoch_test_acc

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, num_classes):
    """训练模型"""
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 更新进度条
            train_pbar.set_postfix(loss=loss.item())
        
        # 计算训练集指标
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        epoch_test_loss, epoch_test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f'Test Loss: {epoch_test_loss:.4f} Acc: {epoch_test_acc:.4f}')
        
        # 保存最佳模型
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with accuracy: {best_acc:.4f}')
    
    # 训练结束后，加载最佳模型并在测试集上进行最终评估
    print(f'\nLoading best model for final evaluation...')
    best_model = resnet50(pretrained=False, num_attributes=num_classes)
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model = best_model.to(device)
    
    final_loss, final_acc = evaluate_model(best_model, test_loader, criterion, device)
    print(f'\nFinal evaluation with best model:')
    print(f'Test Loss: {final_loss:.4f} Acc: {final_acc:.4f}')
    
    print(f'Training complete. Best test accuracy: {best_acc:.4f}')
    return final_acc

def main():
    args = get_args()
    
    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # 加载数据
    train_loader, test_loader, num_classes, class_to_idx = get_data_loaders(
        args.train_path,
        args.test_path,
        args.batch_size,
        args.num_workers
    )
    print(f'Number of classes: {num_classes}')
    print(f'Class mapping: {class_to_idx}')
    
    # 保存类别映射关系，用于后续预测
    with open('class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)
    
    # 初始化模型
    model = resnet50(pretrained=args.pretrained, num_attributes=num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, device, args.epochs, num_classes)

if __name__ == '__main__':
    main()
