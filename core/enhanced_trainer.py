#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 增强训练器
专门支持简化改进版MoE模型的训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
from typing import Dict, Any, Tuple
from tqdm import tqdm

from .base_model import ModelTrainer


class EnhancedModelTrainer(ModelTrainer):
    """增强的模型训练器，专门支持简化改进版MoE"""
    
    def __init__(self, model, device: str = 'cpu'):
        super().__init__(model, device)
        self.expert_usage_history = []
        self.loss_history = {'classification': [], 'load_balance': [], 'diversity': []}
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 20,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 5,
              save_path: str = None) -> Dict[str, Any]:
        """训练模型（支持简化改进版MoE）"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
        
        best_val_acc = 0
        patience_counter = 0
        training_history = {
            'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
            'classification_loss': [], 'load_balance_loss': [], 'diversity_loss': [],
            'expert_usage': [], 'gate_weights_std': []
        }
        
        print(f"🚀 开始训练 {self.model.__class__.__name__}...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            train_metrics = self._train_epoch_enhanced(train_loader, optimizer)
            
            # 验证阶段
            val_metrics = self._validate_epoch_enhanced(val_loader)
            
            # 记录历史
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            training_history['train_loss'].append(train_metrics['total_loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['classification_loss'].append(train_metrics.get('classification_loss', 0))
            training_history['load_balance_loss'].append(train_metrics.get('load_balance_loss', 0))
            training_history['diversity_loss'].append(train_metrics.get('diversity_loss', 0))
            training_history['expert_usage'].append(train_metrics.get('expert_usage', {}))
            training_history['gate_weights_std'].append(train_metrics.get('gate_weights_std', 0))
            
            # 打印训练信息
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            
            # 如果是简化改进版MoE模型，打印详细信息
            if hasattr(self.model, 'compute_total_loss'):
                print(f"  分类损失: {train_metrics.get('classification_loss', 0):.4f}, "
                      f"负载均衡: {train_metrics.get('load_balance_loss', 0):.4f}, "
                      f"多样性: {train_metrics.get('diversity_loss', 0):.4f}")
                
                if 'expert_usage' in train_metrics and train_metrics['expert_usage']:
                    usage_str = ", ".join([f"{k}: {v:.1f}%" for k, v in train_metrics['expert_usage'].items()])
                    print(f"  专家使用率: {usage_str}")
                    print(f"  门控权重标准差: {train_metrics.get('gate_weights_std', 0):.4f}")
            
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
    
    def _train_epoch_enhanced(self, train_loader: DataLoader, optimizer) -> Dict[str, float]:
        """训练一个epoch（增强版）"""
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        total_load_balance_loss = 0
        total_diversity_loss = 0
        correct = 0
        total = 0
        
        # 动态获取专家数量和名称
        if hasattr(self.model, 'expert_names'):
            expert_names = ['cnn', 'lstm', 'mamba', 'transformer']  # 保持原有名称
        else:
            expert_names = ['expert_' + str(i) for i in range(self.model.num_experts if hasattr(self.model, 'num_experts') else 4)]
        
        # 专家使用统计
        expert_usage_counts = {name: 0 for name in expert_names}
        gate_weights_all = []
        
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 检查是否是MoE模型
            if hasattr(self.model, 'compute_total_loss') or hasattr(self.model, 'expert_config'):
                # 处理不同MoE模型的返回格式
                if hasattr(self.model, 'expert_config') or hasattr(self.model, 'expert_names'):
                    # 高级MoE模型或同构MoE模型
                    result = self.model(data, return_gate_weights=True, return_expert_outputs=True)
                    if len(result) == 3:
                        logits, gate_weights, expert_outputs = result
                    else:
                        logits = result[0]
                        gate_weights = result[1] if len(result) > 1 else None
                        expert_outputs = result[2] if len(result) > 2 else None
                else:
                    # 简化改进版MoE模型
                    try:
                        logits, gate_weights, expert_outputs = self.model(data, return_expert_outputs=True)
                    except ValueError:
                        result = self.model(data, return_expert_outputs=True)
                        logits = result[0]
                        gate_weights = result[1] if len(result) > 1 else None
                        expert_outputs = result[2] if len(result) > 2 else None
                
                # 计算总损失
                loss, loss_components = self.model.compute_total_loss(logits, target, gate_weights, expert_outputs)
                
                total_classification_loss += loss_components['classification_loss']
                total_load_balance_loss += loss_components['load_balance_loss']
                total_diversity_loss += loss_components['diversity_loss']
                
                # 统计专家使用情况
                dominant_experts = torch.argmax(gate_weights, dim=1)
                for expert_idx in dominant_experts:
                    if expert_idx < len(expert_names):
                        expert_usage_counts[expert_names[expert_idx]] += 1
                
                # 收集门控权重用于分析
                gate_weights_all.append(gate_weights.detach().cpu())
                
            else:
                # 普通模型
                logits = self.model(data)
                loss = F.cross_entropy(logits, target)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        # 计算专家使用百分比
        expert_usage_percentages = {}
        if total > 0:
            for expert, count in expert_usage_counts.items():
                expert_usage_percentages[expert] = (count / total) * 100
        
        # 计算门控权重标准差
        gate_weights_std = 0
        if gate_weights_all:
            all_weights = torch.cat(gate_weights_all, dim=0)
            gate_weights_std = torch.std(all_weights, dim=0).mean().item()
        
        result = {
            'total_loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
        
        # 添加简化改进版MoE特定指标
        if hasattr(self.model, 'compute_total_loss'):
            result.update({
                'classification_loss': total_classification_loss / len(train_loader),
                'load_balance_loss': total_load_balance_loss / len(train_loader),
                'diversity_loss': total_diversity_loss / len(train_loader),
                'expert_usage': expert_usage_percentages,
                'gate_weights_std': gate_weights_std
            })
        
        return result
    
    def _validate_epoch_enhanced(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch（增强版）"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                # 检查是否是MoE模型
                if hasattr(self.model, 'compute_total_loss') or hasattr(self.model, 'expert_config'):
                    # 处理不同MoE模型的返回格式
                    if hasattr(self.model, 'expert_config') or hasattr(self.model, 'expert_names'):
                        # 高级MoE模型或同构MoE模型
                        result = self.model(data, return_gate_weights=True, return_expert_outputs=True)
                        if len(result) == 3:
                            logits, gate_weights, expert_outputs = result
                        else:
                            logits = result[0]
                            gate_weights = result[1] if len(result) > 1 else None
                            expert_outputs = result[2] if len(result) > 2 else None
                    else:
                        # 简化改进版MoE模型
                        try:
                            logits, gate_weights, expert_outputs = self.model(data, return_expert_outputs=True)
                        except ValueError:
                            result = self.model(data, return_expert_outputs=True)
                            logits = result[0]
                            gate_weights = result[1] if len(result) > 1 else None
                            expert_outputs = result[2] if len(result) > 2 else None
                else:
                    # 普通模型
                    logits = self.model(data)
                
                loss = F.cross_entropy(logits, target)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
    
    def analyze_expert_usage_detailed(self, test_loader: DataLoader) -> Dict[str, Any]:
        """详细分析专家使用情况"""
        if not hasattr(self.model, 'get_gate_weights'):
            return {"message": "模型不支持专家分析"}
        
        print(f"📊 详细分析专家使用情况...")
        
        self.model.eval()
        # 动态获取专家数量和名称
        if hasattr(self.model, 'expert_names'):
            expert_names = self.model.expert_names
        else:
            # 默认专家名称
            expert_names = ['expert_' + str(i) for i in range(self.model.num_experts if hasattr(self.model, 'num_experts') else 4)]
        
        # 专家使用统计
        expert_usage = {name: 0 for name in expert_names}
        expert_accuracy = {name: [] for name in expert_names}
        expert_confidence = {name: [] for name in expert_names}
        gate_weights_history = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Analyzing"):
                data, target = data.to(self.device), target.to(self.device)
                
                gate_weights = self.model.get_gate_weights(data)
                logits = self.model(data)
                pred = logits.argmax(dim=1)
                probs = F.softmax(logits, dim=1)
                
                gate_weights_history.append(gate_weights.cpu().numpy())
                
                # 分析每个样本的专家使用情况
                dominant_experts = torch.argmax(gate_weights, dim=1)
                max_probs = torch.max(probs, dim=1)[0]  # 预测置信度
                
                for i, expert_idx in enumerate(dominant_experts):
                    if expert_idx < len(expert_names):
                        expert_name = expert_names[expert_idx]
                        expert_usage[expert_name] += 1
                        
                        # 记录该专家的准确率和置信度
                        is_correct = (pred[i] == target[i]).item()
                        confidence = max_probs[i].item()
                        
                        expert_accuracy[expert_name].append(is_correct)
                        expert_confidence[expert_name].append(confidence)
        
        # 计算统计信息
        total_samples = sum(expert_usage.values())
        usage_percentages = {}
        accuracy_by_expert = {}
        confidence_by_expert = {}
        
        for expert, count in expert_usage.items():
            usage_percentages[expert] = (count / total_samples) * 100 if total_samples > 0 else 0
            
            if expert_accuracy[expert]:
                accuracy_by_expert[expert] = np.mean(expert_accuracy[expert]) * 100
                confidence_by_expert[expert] = np.mean(expert_confidence[expert])
            else:
                accuracy_by_expert[expert] = 0
                confidence_by_expert[expert] = 0
        
        # 计算门控权重统计
        all_gate_weights = np.concatenate(gate_weights_history, axis=0)
        mean_gate_weights = np.mean(all_gate_weights, axis=0)
        std_gate_weights = np.std(all_gate_weights, axis=0)
        
        # 计算专家使用的均衡性
        usage_values = list(usage_percentages.values())
        usage_balance = 1.0 - (np.std(usage_values) / (np.mean(usage_values) + 1e-8))
        
        analysis_result = {
            'expert_usage_counts': expert_usage,
            'expert_usage_percentages': usage_percentages,
            'expert_accuracy': accuracy_by_expert,
            'expert_confidence': confidence_by_expert,
            'mean_gate_weights': {expert_names[i]: mean_gate_weights[i] for i in range(len(expert_names))},
            'std_gate_weights': {expert_names[i]: std_gate_weights[i] for i in range(len(expert_names))},
            'usage_balance_score': usage_balance,
            'total_samples': total_samples
        }
        
        # 打印详细分析结果
        print(f"\n📈 详细专家使用分析结果:")
        print(f"{'专家':<12} {'使用率':<10} {'准确率':<10} {'置信度':<10} {'平均权重':<12} {'权重标准差':<12}")
        print("-" * 75)
        
        for expert in expert_names:
            usage_pct = usage_percentages[expert]
            accuracy = accuracy_by_expert[expert]
            confidence = confidence_by_expert[expert]
            mean_weight = mean_gate_weights[expert_names.index(expert)]
            std_weight = std_gate_weights[expert_names.index(expert)]
            
            print(f"{expert.upper():<12} {usage_pct:<9.1f}% {accuracy:<9.1f}% {confidence:<9.3f} {mean_weight:<11.3f} {std_weight:<11.3f}")
        
        print(f"\n📊 专家使用均衡性评分: {usage_balance:.3f} (越接近1越均衡)")
        
        return analysis_result
    
    def get_training_insights(self) -> Dict[str, Any]:
        """获取训练过程的洞察"""
        insights = {
            'expert_usage_evolution': self.expert_usage_history,
            'loss_components_evolution': self.loss_history
        }
        
        if self.expert_usage_history:
            # 分析专家使用的演化趋势
            final_usage = self.expert_usage_history[-1] if self.expert_usage_history else {}
            initial_usage = self.expert_usage_history[0] if self.expert_usage_history else {}
            
            usage_change = {}
            for expert in ['cnn', 'lstm', 'mamba', 'transformer']:
                final_pct = final_usage.get(expert, 0)
                initial_pct = initial_usage.get(expert, 0)
                usage_change[expert] = final_pct - initial_pct
            
            insights['usage_change'] = usage_change
        
        return insights