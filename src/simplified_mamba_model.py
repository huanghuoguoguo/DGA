#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 简化版Mamba模型
基于状态空间模型的高效序列建模（简化实现）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
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


class SimplifiedMambaBlock(nn.Module):
    """简化版Mamba块 - 使用RNN和门控机制模拟状态空间模型"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super(SimplifiedMambaBlock, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # 状态空间参数
        self.state_proj = nn.Linear(d_model, d_state)
        self.input_proj = nn.Linear(d_model, d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        
        # 门控机制
        self.gate = nn.Linear(d_model, d_model)
        
        # 归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 状态转移矩阵（可学习）
        self.state_transition = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # 当前输入
            x_t = x[:, t, :]  # (batch, d_model)
            
            # 状态更新
            h_proj = self.state_proj(x_t)  # (batch, d_state)
            x_proj = self.input_proj(x_t)  # (batch, d_state)
            
            # 简化的状态空间更新
            h = torch.tanh(h @ self.state_transition.T + h_proj + x_proj)
            
            # 输出投影
            output_t = self.output_proj(h)  # (batch, d_model)
            
            # 门控
            gate_t = torch.sigmoid(self.gate(x_t))
            output_t = output_t * gate_t
            
            outputs.append(output_t)
        
        # 拼接所有时间步的输出
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        # 残差连接和dropout
        output = self.dropout(output) + residual
        
        return output


class SimplifiedMambaModel(nn.Module):
    """简化版Mamba模型"""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16, 
                 num_classes=2, dropout=0.1):
        super(SimplifiedMambaModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Mamba层
        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(d_model, d_state, dropout)
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
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
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


def load_dataset(file_path='./data/small_dga_dataset.pkl'):
    """加载数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def train_mamba_model():
    """训练简化版Mamba模型"""
    print("=== DGA恶意域名检测 - 简化版Mamba模型训练 ===\\n")
    
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
    
    print(f"\\n数据集划分:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建简化版Mamba模型
    print("\\n创建简化版Mamba模型...")
    model = SimplifiedMambaModel(
        vocab_size=vocab_size, 
        d_model=128,  # 减小模型大小以提高训练速度
        n_layers=3, 
        d_state=16,
        num_classes=2, 
        dropout=0.1
    )
    model.to(device)
    
    model_info = model.get_model_info()
    print(f"简化版Mamba模型参数量: {model_info['total_params']:,}")
    print(f"简化版Mamba模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 训练参数
    num_epochs = 15
    learning_rate = 0.001
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
    
    # 训练循环
    best_val_acc = 0
    patience = 6
    patience_counter = 0
    
    print("开始训练简化版Mamba模型...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), './data/best_simplified_mamba_model.pth')
            print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                break
        
        scheduler.step(val_acc)
    
    training_time = time.time() - start_time
    print(f"\\n简化版Mamba训练完成！用时: {training_time:.2f}秒")
    
    # 测试评估
    model.load_state_dict(torch.load('./data/best_simplified_mamba_model.pth'))
    print(f"\\n在测试集上评估最佳简化版Mamba模型...")
    
    model.eval()
    all_preds = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 测量推理时间
            start_time_inference = time.time()
            output = model(data)
            inference_time = time.time() - start_time_inference
            inference_times.append(inference_time)
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    print(f"\\n=== 简化版Mamba详细评估结果 ===")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"平均推理时间: {np.mean(inference_times)*1000:.2f}ms")
    
    # 分类报告
    print(f"\\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=['Benign', 'Malicious']))
    
    # 保存结果
    results = {
        'model_name': 'Simplified Mamba',
        'model_info': model_info,
        'best_val_accuracy': best_val_acc,
        'test_results': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time_ms': np.mean(inference_times) * 1000
        }
    }
    
    with open('./data/simplified_mamba_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\\n🎉 简化版Mamba训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {accuracy:.4f}")
    print(f"结果已保存到 ./data/simplified_mamba_results.pkl")
    
    return accuracy, model_info


if __name__ == "__main__":
    import math
    train_mamba_model()