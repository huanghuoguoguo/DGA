#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - LSTM模型
基于循环神经网络的序列建模方法
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


class LSTMModel(nn.Module):
    """LSTM模型用于DGA检测"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 注意力机制（兼容旧版PyTorch）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM输出维度
            num_heads=8,
            dropout=dropout
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # 双向LSTM输出维度 * 2
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x):
        # 嵌入: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Layer Normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # 注意力机制（需要转置维度兼容旧版）
        # 转置: (batch_size, seq_len, hidden_dim) -> (seq_len, batch_size, hidden_dim)
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, attn_weights = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        # 转回: (seq_len, batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        attn_out = attn_out.transpose(0, 1)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        # 全连接层
        x = F.relu(self.fc1(pooled))
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


class SimpleLSTM(nn.Module):
    """简单LSTM模型（用于对比）"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        last_output = hidden[-1]  # 取最后一层的隐藏状态
        
        # 全连接
        x = self.dropout(last_output)
        x = self.fc(x)
        
        return x
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }


class DGATrainer:
    """DGA检测模型训练器"""
    
    def __init__(self, model, device='cpu', model_name='LSTM'):
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
    
    def train(self, train_loader, val_loader, num_epochs=30, lr=0.001):
        """训练模型"""
        print(f"开始训练{self.model_name}模型，设备: {self.device}")
        print(f"模型信息: {self.model.get_model_info()}")
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_acc = 0
        patience = 10
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
                safe_model_name = self.model_name.lower().replace(' ', '_').replace('+', '_')
                model_path = f'./data/best_{safe_model_name}_model.pth'
                torch.save(self.model.state_dict(), model_path)
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                    break
        
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
        plt.title(f'{self.model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'./data/{self.model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: ./data/{self.model_name.lower()}_confusion_matrix.png")
        plt.close()
        
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
        ax1.set_title(f'{self.model_name} Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='red')
        ax2.set_title(f'{self.model_name} Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./data/{self.model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        print(f"训练历史已保存到: ./data/{self.model_name.lower()}_training_history.png")
        plt.close()


def load_dataset(file_path='./data/small_dga_dataset.pkl'):
    """加载数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def compare_models(cnn_results, lstm_results, simple_lstm_results):
    """对比模型性能"""
    print("\n" + "="*80)
    print("模型性能对比")
    print("="*80)
    
    models = ['CNN', 'BiLSTM+Attention', 'Simple LSTM']
    results = [cnn_results, lstm_results, simple_lstm_results]
    
    print(f"{'模型':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'推理时间(ms)':<12} {'参数量':<10}")
    print("-" * 90)
    
    for model_name, result in zip(models, results):
        if result:
            print(f"{model_name:<20} {result['test_results']['accuracy']:.4f}     "
                  f"{result['test_results']['precision']:.4f}     "
                  f"{result['test_results']['recall']:.4f}     "
                  f"{result['test_results']['f1']:.4f}     "
                  f"{result['test_results']['inference_time_ms']:.2f}        "
                  f"{result['model_info']['total_params']:,}")
    
    # 找出最佳模型
    best_acc_idx = np.argmax([r['test_results']['accuracy'] if r else 0 for r in results])
    best_f1_idx = np.argmax([r['test_results']['f1'] if r else 0 for r in results])
    fastest_idx = np.argmin([r['test_results']['inference_time_ms'] if r else float('inf') for r in results])
    
    print(f"\n🏆 性能总结:")
    print(f"  最高准确率: {models[best_acc_idx]} ({results[best_acc_idx]['test_results']['accuracy']:.4f})")
    print(f"  最高F1分数: {models[best_f1_idx]} ({results[best_f1_idx]['test_results']['f1']:.4f})")
    print(f"  最快推理: {models[fastest_idx]} ({results[fastest_idx]['test_results']['inference_time_ms']:.2f}ms)")


def main():
    """主函数 - 训练LSTM模型并与CNN对比"""
    print("=== DGA恶意域名检测 - LSTM模型训练与对比 ===\n")
    
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
    
    # 训练两种LSTM模型
    models_to_train = [
        ('BiLSTM+Attention', LSTMModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3)),
        ('Simple LSTM', SimpleLSTM(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3))
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
        trainer = DGATrainer(model, device, model_name)
        
        # 训练模型
        best_val_acc = trainer.train(
            train_loader, val_loader, 
            num_epochs=25, lr=0.001
        )
        
        # 绘制训练历史
        trainer.plot_training_history()
        
        # 加载最佳模型进行测试
        safe_model_name = model_name.lower().replace(' ', '_').replace('+', '_')
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
        result_file = f'./data/{model_name.lower().replace(" ", "_").replace("+", "_")}_results.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        
        all_results[model_name] = results
        
        print(f"\n{model_name}训练完成！")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"测试准确率: {test_results['accuracy']:.4f}")
    
    # 加载CNN结果进行对比
    cnn_results = None
    if os.path.exists('./data/training_results.pkl'):
        with open('./data/training_results.pkl', 'rb') as f:
            cnn_results = pickle.load(f)
        cnn_results['model_name'] = 'CNN'
    
    # 模型对比
    compare_models(
        cnn_results, 
        all_results.get('BiLSTM+Attention'), 
        all_results.get('Simple LSTM')
    )
    
    print(f"\n🎉 所有模型训练完成！")
    print(f"\n生成的文件:")
    for model_name in all_results.keys():
        safe_name = model_name.lower().replace(" ", "_").replace("+", "_")
        print(f"  - {model_name}模型: ./data/best_{safe_name}_model.pth")
        print(f"  - {model_name}结果: ./data/{safe_name}_results.pkl")
        print(f"  - {model_name}训练历史: ./data/{safe_name}_training_history.png")
        print(f"  - {model_name}混淆矩阵: ./data/{safe_name}_confusion_matrix.png")


if __name__ == "__main__":
    main()