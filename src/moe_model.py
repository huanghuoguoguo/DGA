#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - MoE（Mixture of Experts）架构
结合CNN和LSTM专家网络的混合模型
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


class CNNExpert(nn.Module):
    """CNN专家网络"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256):
        super(CNNExpert, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多尺度卷积层
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc = nn.Linear(128 * 3, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        pool1 = F.max_pool1d(conv1_out, kernel_size=conv1_out.size(2)).squeeze(2)
        pool2 = F.max_pool1d(conv2_out, kernel_size=conv2_out.size(2)).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, kernel_size=conv3_out.size(2)).squeeze(2)
        
        features = torch.cat([pool1, pool2, pool3], dim=1)
        expert_output = F.relu(self.fc(features))
        return self.dropout(expert_output)


class LSTMExpert(nn.Module):
    """LSTM专家网络"""
    
    def __init__(self, vocab_size, embed_dim=64, lstm_hidden=128, hidden_dim=256):
        super(LSTMExpert, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc = nn.Linear(lstm_hidden * 2, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        pooled = torch.mean(lstm_out, dim=1)
        expert_output = F.relu(self.fc(pooled))
        return self.dropout(expert_output)


class GatingNetwork(nn.Module):
    """门控网络"""
    
    def __init__(self, vocab_size, embed_dim=64, num_experts=2):
        super(GatingNetwork, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        features = self.feature_extractor(embedded)
        gate_weights = self.gate(features)
        return gate_weights


class DGAMoEModel(nn.Module):
    """DGA检测的MoE模型"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_classes=2):
        super(DGAMoEModel, self).__init__()
        
        self.num_experts = 2
        self.hidden_dim = hidden_dim
        
        # 专家网络
        self.experts = nn.ModuleList([
            CNNExpert(vocab_size, embed_dim, hidden_dim),      # 专家0: CNN
            LSTMExpert(vocab_size, embed_dim, 128, hidden_dim) # 专家1: LSTM
        ])
        
        # 门控网络
        self.gating_network = GatingNetwork(vocab_size, embed_dim, self.num_experts)
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.load_balance_weight = 0.01
        
    def forward(self, x):
        # 获取门控权重
        gate_weights = self.gating_network(x)
        
        # 计算每个专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_weights_expanded = gate_weights.unsqueeze(2)
        
        # 加权融合专家输出
        moe_output = torch.sum(expert_outputs * gate_weights_expanded, dim=1)
        
        # 最终分类
        logits = self.classifier(moe_output)
        
        return logits, gate_weights
    
    def get_load_balance_loss(self, gate_weights):
        """计算负载均衡损失"""
        expert_usage = torch.mean(gate_weights, dim=0)
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.mean((expert_usage - target_usage) ** 2)
        return load_balance_loss
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'num_experts': self.num_experts
        }


class MoETrainer:
    """MoE模型训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.gate_weights_history = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_gate_weights = []
        
        progress_bar = tqdm(train_loader, desc="Training MoE")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            logits, gate_weights = self.model(data)
            classification_loss = criterion(logits, target)
            load_balance_loss = self.model.get_load_balance_loss(gate_weights)
            total_loss_batch = classification_loss + self.model.load_balance_weight * load_balance_loss
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            pred = logits.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            total_gate_weights.append(gate_weights.detach().cpu().numpy())
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Acc': f'{100. * total_correct / total_samples:.2f}%',
                'CNN': f'{gate_weights[:, 0].mean().item():.3f}',
                'LSTM': f'{gate_weights[:, 1].mean().item():.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * total_correct / total_samples
        epoch_gate_weights = np.concatenate(total_gate_weights, axis=0)
        avg_gate_weights = np.mean(epoch_gate_weights, axis=0)
        
        return avg_loss, accuracy, avg_gate_weights
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        total_gate_weights = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating MoE"):
                data, target = data.to(self.device), target.to(self.device)
                
                logits, gate_weights = self.model(data)
                classification_loss = criterion(logits, target)
                load_balance_loss = self.model.get_load_balance_loss(gate_weights)
                total_loss_batch = classification_loss + self.model.load_balance_weight * load_balance_loss
                
                total_loss += total_loss_batch.item()
                pred = logits.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                total_gate_weights.append(gate_weights.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * total_correct / total_samples
        val_gate_weights = np.concatenate(total_gate_weights, axis=0)
        avg_gate_weights = np.mean(val_gate_weights, axis=0)
        
        return avg_loss, accuracy, all_preds, all_targets, avg_gate_weights
    
    def train(self, train_loader, val_loader, num_epochs=25, lr=0.001):
        """训练模型"""
        print(f"开始训练MoE模型，设备: {self.device}")
        print(f"模型信息: {self.model.get_model_info()}")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            train_loss, train_acc, train_gate_weights = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_preds, val_targets, val_gate_weights = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.gate_weights_history.append({
                'epoch': epoch,
                'train_gate_weights': train_gate_weights,
                'val_gate_weights': val_gate_weights
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Train Expert Usage - CNN: {train_gate_weights[0]:.3f}, LSTM: {train_gate_weights[1]:.3f}")
            print(f"Val Expert Usage - CNN: {val_gate_weights[0]:.3f}, LSTM: {val_gate_weights[1]:.3f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), './data/best_moe_model.pth')
                print(f"新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                    break
        
        training_time = time.time() - start_time
        print(f"\nMoE训练完成！用时: {training_time:.2f}秒")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def evaluate_detailed(self, test_loader, class_names=['Benign', 'Malicious']):
        """详细评估模型"""
        self.model.eval()
        all_preds = []
        all_targets = []
        inference_times = []
        all_gate_weights = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = time.time()
                logits, gate_weights = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                pred = logits.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        test_gate_weights = np.concatenate(all_gate_weights, axis=0)
        avg_gate_weights = np.mean(test_gate_weights, axis=0)
        
        print(f"\n=== MoE详细评估结果 ===")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"平均推理时间: {np.mean(inference_times)*1000:.2f}ms")
        print(f"专家使用情况 - CNN: {avg_gate_weights[0]:.3f}, LSTM: {avg_gate_weights[1]:.3f}")
        
        print(f"\n分类报告:")
        print(classification_report(all_targets, all_preds, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time_ms': np.mean(inference_times) * 1000,
            'expert_usage': avg_gate_weights
        }


def load_dataset(file_path='./data/small_dga_dataset.pkl'):
    """加载数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def main():
    """主函数"""
    print("=== DGA恶意域名检测 - MoE架构训练 ===\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if not os.path.exists('./data/small_dga_dataset.pkl'):
        print("数据集文件不存在，请先运行 dataset_builder.py")
        return
    
    dataset = load_dataset('./data/small_dga_dataset.pkl')
    X, y = dataset['X'], dataset['y']
    vocab_size = dataset['vocab_size']
    
    print(f"数据集信息:")
    print(f"  样本数: {len(X)}")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  类别分布: {np.bincount(y)}")
    
    full_dataset = DGADataset(X, y)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n创建MoE模型...")
    model = DGAMoEModel(vocab_size=vocab_size, embed_dim=64, hidden_dim=256, num_classes=2)
    
    model_info = model.get_model_info()
    print(f"MoE模型参数量: {model_info['total_params']:,}")
    print(f"MoE模型大小: {model_info['model_size_mb']:.2f} MB")
    
    trainer = MoETrainer(model, device)
    
    print(f"\n开始训练MoE模型...")
    best_val_acc = trainer.train(train_loader, val_loader, num_epochs=25, lr=0.001)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('./data/best_moe_model.pth'))
    
    print(f"\n在测试集上评估MoE模型...")
    test_results = trainer.evaluate_detailed(test_loader)
    
    # 保存结果
    results = {
        'model_name': 'MoE (CNN+LSTM)',
        'model_info': model_info,
        'best_val_accuracy': best_val_acc,
        'test_results': test_results,
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies,
            'gate_weights_history': trainer.gate_weights_history
        }
    }
    
    with open('./data/moe_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n🎉 MoE训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_results['accuracy']:.4f}")
    print(f"结果已保存到 ./data/moe_results.pkl")


if __name__ == "__main__":
    main()