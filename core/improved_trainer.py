#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 改进的模型训练器
支持MoE模型的负载均衡损失
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
from typing import Dict, Any, Tuple
from tqdm import tqdm

from .base_model import ModelTrainer


class ImprovedModelTrainer(ModelTrainer):
    """改进的模型训练器，支持MoE负载均衡"""
    
    def __init__(self, model, device: str = 'cpu', load_balance_weight: float = 0.1):
        super().__init__(model, device)
        self.load_balance_weight = load_balance_weight
        self.expert_usage_history = []
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 20,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 5,
              save_path: str = None) -> Dict[str, Any]:
        """训练模型（支持MoE负载均衡）"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
        
        best_val_acc = 0
        patience_counter = 0
        training_history = {
            'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
            'load_balance_loss': [], 'expert_usage': []
        }
        
        print(f"🚀 开始训练 {self.model.__class__.__name__}...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            train_metrics = self._train_epoch_improved(train_loader, optimizer, criterion)
            
            # 验证阶段
            val_metrics = self._validate_epoch_improved(val_loader, criterion)
            
            # 记录历史
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['load_balance_loss'].append(train_metrics.get('load_balance_loss', 0))
            training_history['expert_usage'].append(train_metrics.get('expert_usage', {}))
            
            # 打印训练信息
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            
            # 如果是MoE模型，打印专家使用情况
            if 'expert_usage' in train_metrics and train_metrics['expert_usage']:
                usage_str = ", ".join([f"{k}: {v:.1f}%" for k, v in train_metrics['expert_usage'].items()])
                print(f"  专家使用率: {usage_str}")
                if 'load_balance_loss' in train_metrics:
                    print(f"  负载均衡损失: {train_metrics['load_balance_loss']:.4f}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✅ 新的最佳验证准确率: {best_val_acc:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续{patience}个epoch未提升，提前停止训练")
                    break
            
            scheduler.step(val_metrics['accuracy'])
        
        training_time = time.time() - start_time
        print(f"训练完成！用时: {training_time:.2f}秒")
        
        return {
            'best_val_accuracy': best_val_acc,
            'training_time': training_time,
            'training_history': training_history,
            'final_epoch': epoch
        }
    
    def _train_epoch_improved(self, train_loader: DataLoader, optimizer, criterion) -> Dict[str, float]:
        """训练一个epoch（改进版，支持MoE）"""
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        total_load_balance_loss = 0
        correct = 0
        total = 0
        
        # 专家使用统计
        expert_usage_counts = {'cnn': 0, 'lstm': 0, 'mamba': 0, 'transformer': 0}
        
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 检查是否是MoE模型
            if hasattr(self.model, 'get_gate_weights'):
                # MoE模型
                output, gate_weights = self.model(data, return_gate_weights=True)
                
                # 分类损失
                classification_loss = criterion(output, target)
                
                # 负载均衡损失
                load_balance_loss = self.model.compute_load_balance_loss(gate_weights)
                
                # 总损失
                loss = classification_loss + self.load_balance_weight * load_balance_loss
                
                total_classification_loss += classification_loss.item()
                total_load_balance_loss += load_balance_loss.item()
                
                # 统计专家使用情况
                dominant_experts = torch.argmax(gate_weights, dim=1)
                expert_names = ['cnn', 'lstm', 'mamba', 'transformer']
                for expert_idx in dominant_experts:
                    if expert_idx < len(expert_names):
                        expert_usage_counts[expert_names[expert_idx]] += 1
            else:
                # 普通模型
                output = self.model(data)
                loss = criterion(output, target)
                classification_loss = loss
                load_balance_loss = torch.tensor(0.0)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        # 计算专家使用百分比
        expert_usage_percentages = {}
        if total > 0:
            for expert, count in expert_usage_counts.items():
                expert_usage_percentages[expert] = (count / total) * 100
        
        result = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
        
        # 添加MoE特定指标
        if hasattr(self.model, 'get_gate_weights'):
            result['classification_loss'] = total_classification_loss / len(train_loader)
            result['load_balance_loss'] = total_load_balance_loss / len(train_loader)
            result['expert_usage'] = expert_usage_percentages
        
        return result
    
    def _validate_epoch_improved(self, val_loader: DataLoader, criterion) -> Dict[str, float]:
        """验证一个epoch（改进版）"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                if hasattr(self.model, 'get_gate_weights'):
                    output = self.model(data)  # 验证时不需要gate_weights
                else:
                    output = self.model(data)
                
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
    
    def analyze_expert_usage(self, test_loader: DataLoader) -> Dict[str, Any]:
        """分析专家使用情况"""
        if not hasattr(self.model, 'get_gate_weights'):
            return {"message": "模型不支持专家分析"}
        
        print(f"📊 分析专家使用情况...")
        
        self.model.eval()
        expert_usage = {'cnn': 0, 'lstm': 0, 'mamba': 0, 'transformer': 0}
        expert_accuracy = {'cnn': [], 'lstm': [], 'mamba': [], 'transformer': []}
        gate_weights_history = []
        
        expert_names = ['cnn', 'lstm', 'mamba', 'transformer']
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Analyzing"):
                data, target = data.to(self.device), target.to(self.device)
                
                gate_weights = self.model.get_gate_weights(data)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                gate_weights_history.append(gate_weights.cpu().numpy())
                
                # 分析每个样本的专家使用情况
                dominant_experts = torch.argmax(gate_weights, dim=1)
                
                for i, expert_idx in enumerate(dominant_experts):
                    if expert_idx < len(expert_names):
                        expert_name = expert_names[expert_idx]
                        expert_usage[expert_name] += 1
                        
                        # 记录该专家的准确率
                        is_correct = (pred[i] == target[i]).item()
                        expert_accuracy[expert_name].append(is_correct)
        
        # 计算统计信息
        total_samples = sum(expert_usage.values())
        usage_percentages = {}
        accuracy_by_expert = {}
        
        for expert, count in expert_usage.items():
            usage_percentages[expert] = (count / total_samples) * 100 if total_samples > 0 else 0
            if expert_accuracy[expert]:
                accuracy_by_expert[expert] = np.mean(expert_accuracy[expert]) * 100
            else:
                accuracy_by_expert[expert] = 0
        
        # 计算门控权重统计
        all_gate_weights = np.concatenate(gate_weights_history, axis=0)
        mean_gate_weights = np.mean(all_gate_weights, axis=0)
        std_gate_weights = np.std(all_gate_weights, axis=0)
        
        analysis_result = {
            'expert_usage_counts': expert_usage,
            'expert_usage_percentages': usage_percentages,
            'expert_accuracy': accuracy_by_expert,
            'mean_gate_weights': {expert_names[i]: mean_gate_weights[i] for i in range(len(expert_names))},
            'std_gate_weights': {expert_names[i]: std_gate_weights[i] for i in range(len(expert_names))},
            'total_samples': total_samples
        }
        
        # 打印分析结果
        print(f"\n📈 专家使用分析结果:")
        print(f"{'专家':<12} {'使用率':<10} {'准确率':<10} {'平均权重':<12} {'权重标准差':<12}")
        print("-" * 60)
        
        for expert in expert_names:
            usage_pct = usage_percentages[expert]
            accuracy = accuracy_by_expert[expert]
            mean_weight = mean_gate_weights[expert_names.index(expert)]
            std_weight = std_gate_weights[expert_names.index(expert)]
            
            print(f"{expert.upper():<12} {usage_pct:<9.1f}% {accuracy:<9.1f}% {mean_weight:<11.3f} {std_weight:<11.3f}")
        
        return analysis_result