#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 简单CNN模型
用于基线实验和概念验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


class DGADataset(Dataset):
    """DGA数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleCNN(nn.Module):
    """简单的CNN模型用于DGA检测"""
    
    def __init__(self, vocab_size, embed_dim=64, num_classes=2, max_length=40):
        super(SimpleCNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多尺度卷积层
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 全局最大池化后的特征维度
        self.fc_input_dim = 128 * 3  # 3个卷积层的输出
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 嵌入: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # 转置用于卷积: (batch_size, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # 多尺度卷积
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        # 全局最大池化
        pool1 = F.max_pool1d(conv1_out, kernel_size=conv1_out.size(2)).squeeze(2)
        pool2 = F.max_pool1d(conv2_out, kernel_size=conv2_out.size(2)).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, kernel_size=conv3_out.size(2)).squeeze(2)
        
        # 拼接特征
        features = torch.cat([pool1, pool2, pool3], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024  # 假设float32
        }


class DGATrainer:
    """DGA检测模型训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * total_correct / total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * total_correct / total_samples
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, train_loader, val_loader, num_epochs=20, lr=0.001):
        """训练模型"""
        print(f"开始训练，设备: {self.device}")
        print(f"模型信息: {self.model.get_model_info()}")
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        best_val_acc = 0
        patience = 7
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, criterion)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), './data/best_cnn_model.pth')
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                    break
        
        training_time = time.time() - start_time
        print(f"\n训练完成！用时: {training_time:.2f}秒")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def evaluate_detailed(self, test_loader, class_names=['Benign', 'Malicious']):
        """详细评估模型"""
        self.model.eval()
        all_preds = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                output = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        print(f"\n=== 详细评估结果 ===")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"平均推理时间: {np.mean(inference_times)*1000:.2f}ms")
        
        # 分类报告
        print(f"\n分类报告:")
        print(classification_report(all_targets, all_preds, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./data/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\u6df7淆矩阵已保存到: ./data/confusion_matrix.png")
        plt.close()  # 关闭图形以释放内存
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time_ms': np.mean(inference_times) * 1000
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./data/training_history.png', dpi=300, bbox_inches='tight')
        print(f"\u8bad\u7ec3\u5386\u53f2\u5df2\u4fdd\u5b58\u5230: ./data/training_history.png")
        plt.close()  # 关闭图形以释放内存


def load_dataset(file_path='./data/small_dga_dataset.pkl'):
    """加载数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def main():
    """主函数 - 训练简单CNN模型"""
    print("=== DGA恶意域名检测 - 简单CNN训练 ===\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    if not os.path.exists('./data/small_dga_dataset.pkl'):
        print("数据集文件不存在，请先运行 dataset_builder.py")
        return
    
    dataset = load_dataset('./data/small_dga_dataset.pkl')
    X, y = dataset['X'], dataset['y']
    vocab_size = dataset['vocab_size']
    max_length = dataset['max_length']
    
    print(f"数据集信息:")
    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {X.shape}")
    print(f"  类别分布: {np.bincount(y)}")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  最大长度: {max_length}")
    
    # 创建数据集
    full_dataset = DGADataset(X, y)
    
    # 数据集划分 (70% 训练, 15% 验证, 15% 测试)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n创建CNN模型...")
    model = SimpleCNN(
        vocab_size=vocab_size,
        embed_dim=64,
        num_classes=2,
        max_length=max_length
    )
    
    model_info = model.get_model_info()
    print(f"模型参数量: {model_info['total_params']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 创建训练器
    trainer = DGATrainer(model, device)
    
    # 训练模型
    print(f"\n开始训练...")
    best_val_acc = trainer.train(
        train_loader, val_loader, 
        num_epochs=30, lr=0.001
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('./data/best_cnn_model.pth'))
    
    # 详细评估
    print(f"\n在测试集上评估最佳模型...")
    test_results = trainer.evaluate_detailed(test_loader)
    
    # 保存结果
    results = {
        'model_info': model_info,
        'best_val_accuracy': best_val_acc,
        'test_results': test_results,
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies
        }
    }
    
    with open('./data/training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n训练完成！结果已保存到 ./data/training_results.pkl")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_results['accuracy']:.4f}")


if __name__ == "__main__":
    main()