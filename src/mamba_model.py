#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - Mamba模型
基于状态空间模型(SSM)的高效序列建模方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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


class SelectiveScan(nn.Module):
    """选择性扫描机制 - Mamba的核心组件"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super(SelectiveScan, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand_factor * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 卷积层（用于局部上下文）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # Δ和B
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # A参数（可学习的对角矩阵）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D参数（跳跃连接）
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # 各自 (batch, seq_len, d_inner)
        
        # 1D卷积（处理局部依赖）
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # 移除padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # 激活
        x = F.silu(x)
        
        # SSM参数计算
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 计算Δ和B
        x_dbl = self.x_proj(x)  # (batch, seq_len, d_state * 2)
        delta, B = x_dbl.chunk(2, dim=-1)  # 各自 (batch, seq_len, d_state)
        
        # 计算时间步长
        delta = F.softplus(self.dt_proj(x))  # (batch, seq_len, d_inner)
        
        # 选择性状态空间扫描
        y = self.selective_scan(x, delta, A, B)
        
        # 门控机制
        y = y * F.silu(z)
        
        # 输出投影
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, u, delta, A, B):
        """
        选择性扫描算法
        u: (batch, seq_len, d_inner)
        delta: (batch, seq_len, d_inner)
        A: (d_inner, d_state)
        B: (batch, seq_len, d_state)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # 离散化A和B
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (batch, seq_len, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (batch, seq_len, d_inner, d_state)
        
        # 初始化状态
        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        # 序列扫描
        ys = []
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.sum(x, dim=-1)  # (batch, d_inner)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)
        
        # 跳跃连接
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock(nn.Module):
    """Mamba块"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveScan(d_model, d_state, d_conv, expand_factor)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaModel(nn.Module):
    """基于Mamba的DGA检测模型"""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16, 
                 d_conv=4, expand_factor=2, num_classes=2, dropout=0.1):
        super(MambaModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand_factor, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        # 嵌入
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # Mamba层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'n_layers': len(self.layers),
            'd_model': self.d_model
        }


class MambaTrainer:
    """Mamba模型训练器"""
    
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
        
        progress_bar = tqdm(train_loader, desc="Training Mamba")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
            for data, target in tqdm(val_loader, desc="Validating Mamba"):
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
        print(f"开始训练Mamba模型，设备: {self.device}")
        print(f"模型信息: {self.model.get_model_info()}")
        
        # 优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*3, steps_per_epoch=len(train_loader), 
            epochs=num_epochs, pct_start=0.1
        )
        
        best_val_acc = 0
        patience = 8
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, criterion)
            
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
                torch.save(self.model.state_dict(), './data/best_mamba_model.pth')
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                    break
            
            # 更新学习率
            scheduler.step()
        
        training_time = time.time() - start_time
        print(f"\nMamba训练完成！用时: {training_time:.2f}秒")
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
        
        print(f"\n=== Mamba详细评估结果 ===")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"平均推理时间: {np.mean(inference_times)*1000:.2f}ms")
        
        # 分类报告
        print(f"\n分类报告:")
        print(classification_report(all_targets, all_preds, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time_ms': np.mean(inference_times) * 1000
        }


def load_dataset(file_path='./data/small_dga_dataset.pkl'):
    """加载数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def main():
    """主函数 - 训练Mamba模型"""
    print("=== DGA恶意域名检测 - Mamba模型训练 ===\n")
    
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
    
    # 创建Mamba模型
    print("\n创建Mamba模型...")
    model = MambaModel(
        vocab_size=vocab_size, 
        d_model=256, 
        n_layers=4, 
        d_state=16,
        d_conv=4,
        expand_factor=2,
        num_classes=2, 
        dropout=0.1
    )
    
    model_info = model.get_model_info()
    print(f"Mamba模型参数量: {model_info['total_params']:,}")
    print(f"Mamba模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 创建训练器
    trainer = MambaTrainer(model, device)
    
    # 训练模型
    best_val_acc = trainer.train(
        train_loader, val_loader, 
        num_epochs=20, lr=0.001
    )
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('./data/best_mamba_model.pth'))
    
    # 详细评估
    print(f"\n在测试集上评估最佳Mamba模型...")
    test_results = trainer.evaluate_detailed(test_loader)
    
    # 保存结果
    results = {
        'model_name': 'Mamba',
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
    
    # 保存结果
    with open('./data/mamba_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n🎉 Mamba训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_results['accuracy']:.4f}")
    print(f"结果已保存到 ./data/mamba_results.pkl")


if __name__ == "__main__":
    main()