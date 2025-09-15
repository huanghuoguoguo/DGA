#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版MoE训练器
支持负载均衡损失和多样性损失
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base_model import ModelTrainer


class EnhancedMoETrainer(ModelTrainer):
    """增强版MoE训练器"""
    
    def __init__(self, model, device, load_balance_weight=0.01, diversity_weight=0.01):
        super().__init__(model, device)
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 20, learning_rate: float = 0.001, 
              weight_decay: float = 1e-4, patience: int = 5, 
              save_path: str = None) -> Dict[str, Any]:
        """
        训练模型
        """
        print(f"🚀 开始训练 {self.model.__class__.__name__}...")
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # 训练历史
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                output = self.model(data)
                
                # 基础分类损失
                classification_loss = criterion(output, target)
                
                # 总损失
                total_loss = classification_loss
                
                # 添加MoE特有的损失
                if hasattr(self.model, 'get_load_balance_loss'):
                    load_balance_loss = self.model.get_load_balance_loss()
                    total_loss += load_balance_loss
                
                if hasattr(self.model, 'get_diversity_loss'):
                    diversity_loss = self.model.get_diversity_loss()
                    total_loss += diversity_loss
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 统计
                train_loss += total_loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # 更新进度条
                current_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            # 计算训练指标
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_accuracy = 100. * train_correct / train_total
            
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_accuracy)
            
            # 验证阶段
            val_results = self.evaluate(val_loader)
            epoch_val_loss = val_results['loss']
            epoch_val_accuracy = val_results['accuracy'] * 100
            
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
            
            # 学习率调度
            scheduler.step(epoch_val_accuracy)
            
            # 打印epoch结果
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Acc: {epoch_train_accuracy:.2f}%, "
                  f"Val Acc: {epoch_val_accuracy:.2f}%, "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}")
            
            # 保存最佳模型
            if epoch_val_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_val_accuracy
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✅ 新的最佳验证准确率: {best_val_accuracy:.2f}%，模型已保存")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⏹️ 早停：验证准确率连续{patience}轮未改善")
                    break
            
            # 打印专家使用统计（如果可用）
            if hasattr(self.model, 'get_expert_usage_stats') and (epoch + 1) % 5 == 0:
                stats = self.model.get_expert_usage_stats()
                if stats:
                    print(f"  📊 专家使用率: {[f'{p:.1f}%' for p in stats['expert_usage_percentages']]}")
                    print(f"  📊 使用方差: {stats['usage_variance']:.4f}")
        
        training_time = time.time() - start_time
        print(f"训练完成！用时: {training_time:.2f}秒")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'training_time': training_time,
            'final_lr': optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        inference_times = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            eval_pbar = tqdm(data_loader, desc="Evaluating")
            for data, target in eval_pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                output = self.model(data)
                inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)
                
                # 计算损失
                loss = criterion(output, target)
                
                # 添加MoE特有的损失
                if hasattr(self.model, 'get_load_balance_loss'):
                    load_balance_loss = self.model.get_load_balance_loss()
                    loss += load_balance_loss
                
                if hasattr(self.model, 'get_diversity_loss'):
                    diversity_loss = self.model.get_diversity_loss()
                    loss += diversity_loss
                
                total_loss += loss.item()
                
                # 预测
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        avg_loss = total_loss / len(data_loader)
        avg_inference_time = np.mean(inference_times)
        
        print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  平均推理时间: {avg_inference_time:.2f}ms")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'loss': avg_loss,
            'avg_inference_time': avg_inference_time
        }
    
    def analyze_expert_usage_detailed(self, data_loader: DataLoader) -> Optional[Dict[str, Any]]:
        """
        详细分析专家使用情况
        """
        if not hasattr(self.model, 'get_expert_usage_stats'):
            return None
        
        self.model.eval()
        
        # 重置统计
        if hasattr(self.model, 'expert_usage_count'):
            self.model.expert_usage_count.zero_()
            self.model.total_samples = 0
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Analyzing expert usage"):
                data, target = data.to(self.device), target.to(self.device)
                _ = self.model(data)
        
        # 获取统计信息
        stats = self.model.get_expert_usage_stats()
        
        if stats:
            print(f"\n🔍 专家使用分析:")
            for i, usage in enumerate(stats['expert_usage_percentages']):
                print(f"  专家 {i}: {usage:.2f}%")
            print(f"  使用方差: {stats['usage_variance']:.4f}")
            print(f"  总样本数: {stats['total_samples']}")
            
            # 计算负载均衡指标
            usage_array = np.array(stats['expert_usage_percentages'])
            balance_score = 1.0 - (np.std(usage_array) / np.mean(usage_array))
            print(f"  负载均衡分数: {balance_score:.4f} (越接近1越均衡)")
            
            stats['balance_score'] = balance_score
        
        return stats