#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 包含Mamba专家的扩展MoE架构
使用CNN、LSTM、Transformer和Mamba四个专家的混合专家模型
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

# 导入各个专家模型的组件
import sys
sys.path.append('./src')

from mamba_model import SelectiveScan, MambaBlock


class DGADataset(Dataset):
    """DGA数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# CNN专家
class CNNExpert(nn.Module):
    """CNN专家模型"""
    
    def __init__(self, vocab_size, embed_dim=64, num_classes=2, max_length=40):
        super(CNNExpert, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多尺度卷积层
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc = nn.Linear(128 * 3, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq)
        
        # 多尺度卷积
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        # 全局最大池化
        pool1 = F.max_pool1d(conv1_out, kernel_size=conv1_out.size(2)).squeeze(2)
        pool2 = F.max_pool1d(conv2_out, kernel_size=conv2_out.size(2)).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, kernel_size=conv3_out.size(2)).squeeze(2)
        
        # 特征拼接
        features = torch.cat([pool1, pool2, pool3], dim=1)
        features = self.dropout(features)
        
        # 分类
        output = self.fc(features)
        return output


# LSTM专家
class LSTMExpert(nn.Module):
    """LSTM专家模型"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMExpert, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           dropout=dropout, bidirectional=True, batch_first=True)
        
        # 注意力机制
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden_dim*2)
        
        # 注意力机制
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_dim*2)
        
        # 分类
        output = self.classifier(attended)
        return output


# Transformer专家（简化版）
class TransformerExpert(nn.Module):
    """Transformer专家模型（简化版）"""
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, 
                 num_classes=2, max_len=60, dropout=0.1):
        super(TransformerExpert, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 简化的Transformer - 使用PyTorch内置的TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        embedded = self.dropout(embedded)
        
        # Transformer编码
        transformer_out = self.transformer(embedded)  # (batch, seq, d_model)
        
        # 全局平均池化
        pooled = torch.mean(transformer_out, dim=1)  # (batch, d_model)
        
        # 分类
        output = self.classifier(pooled)
        return output


# Mamba专家
class MambaExpert(nn.Module):
    """Mamba专家模型"""
    
    def __init__(self, vocab_size, d_model=256, n_layers=2, d_state=16, 
                 d_conv=4, expand_factor=2, num_classes=2, dropout=0.1):
        super(MambaExpert, self).__init__()
        
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
    
    def forward(self, x):
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


# 门控网络
class AdvancedGatingNetwork(nn.Module):
    """高级门控网络 - 支持四个专家"""
    
    def __init__(self, vocab_size, embed_dim=64, num_experts=4, dropout=0.1):
        super(AdvancedGatingNetwork, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多层特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 门控决策网络
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_experts)
        )
        
    def forward(self, x):
        # x: (batch, seq_len) - 原始输入序列
        embedded = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        features = self.feature_extractor(embedded)  # (batch, 256)
        gate_logits = self.gate(features)  # (batch, num_experts)
        gate_weights = F.softmax(gate_logits, dim=1)
        return gate_weights


# 超级MoE模型
class SuperMoEModel(nn.Module):
    """超级专家混合模型（包含CNN、LSTM、Transformer、Mamba四个专家）"""
    
    def __init__(self, vocab_size, embed_dim=64, num_classes=2, max_length=40):
        super(SuperMoEModel, self).__init__()
        
        self.num_experts = 4
        self.vocab_size = vocab_size
        
        # 创建四个专家
        self.cnn_expert = CNNExpert(vocab_size, embed_dim, num_classes, max_length)
        self.lstm_expert = LSTMExpert(vocab_size, embed_dim, 128, 2, num_classes, 0.3)
        self.transformer_expert = TransformerExpert(vocab_size, 128, 8, 2, num_classes, max_length, 0.1)
        self.mamba_expert = MambaExpert(vocab_size, 256, 2, 16, 4, 2, num_classes, 0.1)
        
        # 高级门控网络
        self.gating_network = AdvancedGatingNetwork(vocab_size, embed_dim, self.num_experts)
        
        # 负载均衡损失的权重
        self.load_balance_weight = 0.01
        
    def forward(self, x, return_expert_outputs=False):
        batch_size = x.size(0)
        
        # 获取门控权重
        gate_weights = self.gating_network(x)  # (batch, 4)
        
        # 获取各专家的输出
        cnn_output = self.cnn_expert(x)
        lstm_output = self.lstm_expert(x)
        transformer_output = self.transformer_expert(x)
        mamba_output = self.mamba_expert(x)
        
        # 专家输出堆叠
        expert_outputs = torch.stack([cnn_output, lstm_output, transformer_output, mamba_output], dim=2)  # (batch, num_classes, num_experts)
        
        # 根据门控权重加权求和
        gate_weights = gate_weights.unsqueeze(1)  # (batch, 1, num_experts)
        final_output = torch.sum(expert_outputs * gate_weights, dim=2)  # (batch, num_classes)
        
        if return_expert_outputs:
            return final_output, gate_weights.squeeze(1), expert_outputs
        
        return final_output, gate_weights.squeeze(1)
    
    def load_balance_loss(self, gate_weights):
        """计算负载均衡损失"""
        # 计算每个专家被选择的平均概率
        expert_usage = torch.mean(gate_weights, dim=0)  # (num_experts,)
        
        # 理想情况下每个专家被选择的概率应该相等
        target_usage = 1.0 / self.num_experts
        
        # 计算不平衡程度
        load_loss = torch.sum((expert_usage - target_usage) ** 2)
        
        return load_loss
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 各专家参数量
        cnn_params = sum(p.numel() for p in self.cnn_expert.parameters())
        lstm_params = sum(p.numel() for p in self.lstm_expert.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_expert.parameters())
        mamba_params = sum(p.numel() for p in self.mamba_expert.parameters())
        gate_params = sum(p.numel() for p in self.gating_network.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'num_experts': self.num_experts,
            'cnn_params': cnn_params,
            'lstm_params': lstm_params,
            'transformer_params': transformer_params,
            'mamba_params': mamba_params,
            'gate_params': gate_params
        }


def load_dataset(file_path='./data/small_dga_dataset.pkl'):
    """加载数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def main():
    """主函数 - 训练超级MoE模型"""
    print("=== DGA恶意域名检测 - 超级MoE架构训练（CNN+LSTM+Transformer+Mamba） ===\\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset('./data/small_dga_dataset.pkl')
    X, y = dataset['X'], dataset['y']
    vocab_size = dataset['vocab_size']
    
    # 创建数据集
    full_dataset = DGADataset(X, y)
    
    # 数据集划分
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建超级MoE模型
    model = SuperMoEModel(vocab_size=vocab_size, embed_dim=64, num_classes=2, max_length=40)
    model.to(device)
    
    model_info = model.get_model_info()
    print(f"超级MoE模型参数量: {model_info['total_params']:,}")
    print(f"CNN专家: {model_info['cnn_params']:,}, LSTM专家: {model_info['lstm_params']:,}")
    print(f"Transformer专家: {model_info['transformer_params']:,}, Mamba专家: {model_info['mamba_params']:,}")
    print(f"门控网络: {model_info['gate_params']:,}")
    
    # 训练参数
    num_epochs = 25
    learning_rate = 0.001
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 训练循环
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print("开始训练超级MoE模型...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_gate_weights = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, gate_weights = model(data)
            
            # 分类损失 + 负载均衡损失
            classification_loss = criterion(output, target)
            load_balance_loss = model.load_balance_loss(gate_weights)
            total_loss = classification_loss + model.load_balance_weight * load_balance_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            epoch_gate_weights.append(gate_weights.detach().cpu())
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_gate_weights = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                data, target = data.to(device), target.to(device)
                output, gate_weights = model(data)
                
                classification_loss = criterion(output, target)
                load_balance_loss = model.load_balance_loss(gate_weights)
                total_loss = classification_loss + model.load_balance_weight * load_balance_loss
                
                val_loss += total_loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                val_gate_weights.append(gate_weights.cpu())
        
        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 专家使用率
        train_expert_usage = torch.mean(torch.cat(epoch_gate_weights, dim=0), dim=0)
        val_expert_usage = torch.mean(torch.cat(val_gate_weights, dim=0), dim=0)
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print(f"Expert Usage - CNN: {train_expert_usage[0]:.3f}, LSTM: {train_expert_usage[1]:.3f}, "
              f"Transformer: {train_expert_usage[2]:.3f}, Mamba: {train_expert_usage[3]:.3f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), './data/best_super_moe_model.pth')
            print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                break
        
        scheduler.step(val_acc)
    
    # 测试评估
    model.load_state_dict(torch.load('./data/best_super_moe_model.pth'))
    print(f"\\n在测试集上评估最佳超级MoE模型...")
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_gate_weights = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, gate_weights = model(data)
            
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
            
            test_gate_weights.append(gate_weights.cpu())
    
    test_acc = 100. * test_correct / test_total
    test_expert_usage = torch.mean(torch.cat(test_gate_weights, dim=0), dim=0)
    
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试集专家使用率 - CNN: {test_expert_usage[0]:.3f}, LSTM: {test_expert_usage[1]:.3f}, "
          f"Transformer: {test_expert_usage[2]:.3f}, Mamba: {test_expert_usage[3]:.3f}")
    
    # 保存结果
    results = {
        'model_name': 'Super MoE (CNN+LSTM+Transformer+Mamba)',
        'model_info': model_info,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc / 100,
        'test_expert_usage': {
            'cnn': test_expert_usage[0].item(),
            'lstm': test_expert_usage[1].item(),
            'transformer': test_expert_usage[2].item(),
            'mamba': test_expert_usage[3].item()
        }
    }
    
    with open('./data/super_moe_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\\n🎉 超级MoE训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"结果已保存到 ./data/super_moe_results.pkl")


if __name__ == "__main__":
    main()