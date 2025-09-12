#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - Transformer模型
基于自注意力机制的序列建模方法
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
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


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # 自注意力
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attn_weights


class BasicTransformer(nn.Module):
    """基础Transformer模型"""
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, 
                 num_classes=2, max_len=60, dropout=0.1):
        super(BasicTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model*4, dropout) 
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        x = x.transpose(0, 1)  # (seq, batch, d_model) for transformer
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 通过Transformer层
        attn_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attn_weights_list.append(attn_weights)
        
        # 全局池化和分类
        x = x.transpose(0, 1)  # (batch, seq, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        logits = self.classifier(x)
        
        return logits
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }


class TransformerCNN(nn.Module):
    """Transformer + CNN融合模型"""
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, 
                 num_classes=2, max_len=60, dropout=0.1):
        super(TransformerCNN, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer分支
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model*2, dropout) 
            for _ in range(num_layers)
        ])
        
        # CNN分支 - 直接作用在embedding上
        self.cnn_conv1 = nn.Conv1d(d_model, 64, kernel_size=2, padding=1)
        self.cnn_conv2 = nn.Conv1d(d_model, 64, kernel_size=3, padding=1)
        self.cnn_conv3 = nn.Conv1d(d_model, 64, kernel_size=4, padding=2)
        
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_bn2 = nn.BatchNorm1d(64)
        self.cnn_bn3 = nn.BatchNorm1d(64)
        
        # 特征融合
        transformer_dim = d_model
        cnn_dim = 64 * 3  # 3个卷积层的输出
        
        self.fusion = nn.Sequential(
            nn.Linear(transformer_dim + cnn_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # 嵌入和位置编码
        embedded = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        
        # Transformer分支
        transformer_input = embedded.transpose(0, 1)  # (seq, batch, d_model)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_input = self.dropout(transformer_input)
        
        transformer_output = transformer_input
        for transformer_block in self.transformer_blocks:
            transformer_output, _ = transformer_block(transformer_output)
        
        transformer_output = transformer_output.transpose(0, 1)  # (batch, seq, d_model)
        transformer_features = torch.mean(transformer_output, dim=1)  # 全局平均池化
        
        # CNN分支 - 作用在embedding上
        cnn_input = embedded.transpose(1, 2)  # (batch, d_model, seq)
        
        conv1_out = F.relu(self.cnn_bn1(self.cnn_conv1(cnn_input)))
        conv2_out = F.relu(self.cnn_bn2(self.cnn_conv2(cnn_input)))
        conv3_out = F.relu(self.cnn_bn3(self.cnn_conv3(cnn_input)))
        
        # 全局最大池化
        pool1 = F.max_pool1d(conv1_out, kernel_size=conv1_out.size(2)).squeeze(2)
        pool2 = F.max_pool1d(conv2_out, kernel_size=conv2_out.size(2)).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, kernel_size=conv3_out.size(2)).squeeze(2)
        
        cnn_features = torch.cat([pool1, pool2, pool3], dim=1)
        
        # 特征融合
        fused_features = torch.cat([transformer_features, cnn_features], dim=1)
        logits = self.fusion(fused_features)
        
        return logits
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }


class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, model, device='cpu', model_name='Transformer'):
        self.model = model
        self.device = device
        self.model_name = model_name
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
        
        progress_bar = tqdm(train_loader, desc=f"Training {self.model_name}")
        
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
            for data, target in tqdm(val_loader, desc=f"Validating {self.model_name}"):
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
    
    def train(self, train_loader, val_loader, num_epochs=25, lr=0.0005):
        """训练模型"""
        print(f"开始训练{self.model_name}模型，设备: {self.device}")
        print(f"模型信息: {self.model.get_model_info()}")
        
        # 优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器 - Transformer通常需要warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*2, steps_per_epoch=len(train_loader), 
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
                # 保存最佳模型
                safe_model_name = self.model_name.lower().replace(' ', '_').replace('+', '_')
                model_path = f'./data/best_{safe_model_name}_model.pth'
                torch.save(self.model.state_dict(), model_path)
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                    break
            
            # 更新学习率
            scheduler.step()
        
        training_time = time.time() - start_time
        print(f"\n{self.model_name}训练完成！用时: {training_time:.2f}秒")
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
        
        print(f"\n=== {self.model_name}详细评估结果 ===")
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
    """主函数 - 训练Transformer模型"""
    print("=== DGA恶意域名检测 - Transformer模型训练 ===\n")
    
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
    
    # 训练两种Transformer模型
    models_to_train = [
        ('Basic Transformer', BasicTransformer(
            vocab_size=vocab_size, d_model=128, nhead=8, num_layers=4, 
            num_classes=2, max_len=max_length, dropout=0.1
        )),
        ('Transformer-CNN', TransformerCNN(
            vocab_size=vocab_size, d_model=128, nhead=8, num_layers=2, 
            num_classes=2, max_len=max_length, dropout=0.1
        ))
    ]
    
    all_results = {}
    
    for model_name, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"训练 {model_name} 模型")
        print(f"{'='*60}")
        
        model_info = model.get_model_info()
        print(f"模型参数量: {model_info['total_params']:,}")
        print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
        
        # 创建训练器
        trainer = TransformerTrainer(model, device, model_name)
        
        # 训练模型
        best_val_acc = trainer.train(
            train_loader, val_loader, 
            num_epochs=20, lr=0.0005 if 'Basic' in model_name else 0.001
        )
        
        # 加载最佳模型进行测试
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        model_file = f'./data/best_{safe_model_name}_model.pth'
        model.load_state_dict(torch.load(model_file))
        
        # 详细评估
        print(f"\n在测试集上评估最佳{model_name}模型...")
        test_results = trainer.evaluate_detailed(test_loader)
        
        # 保存结果
        results = {
            'model_name': model_name,
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
        result_file = f'./data/{safe_model_name}_results.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        
        all_results[model_name] = results
        
        print(f"\n{model_name}训练完成！")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"测试准确率: {test_results['accuracy']:.4f}")
    
    print(f"\n🎉 所有Transformer模型训练完成！")
    print(f"\n生成的文件:")
    for model_name in all_results.keys():
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        print(f"  - {model_name}模型: ./data/best_{safe_name}_model.pth")
        print(f"  - {model_name}结果: ./data/{safe_name}_results.pkl")


if __name__ == "__main__":
    main()