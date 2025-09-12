#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 模型基类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseModel(nn.Module, ABC):
    """所有DGA检测模型的基类"""
    
    def __init__(self, vocab_size: int, num_classes: int = 2):
        super(BaseModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x):
        """前向传播，子类必须实现"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"📋 模型信息:")
        print(f"  总参数量: {info['total_params']:,}")
        print(f"  可训练参数: {info['trainable_params']:,}")
        print(f"  模型大小: {info['model_size_mb']:.2f} MB")
        print(f"  词汇表大小: {info['vocab_size']}")
        print(f"  类别数: {info['num_classes']}")


class ModelTrainer:
    """统一的模型训练器"""
    
    def __init__(self, model: BaseModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 20,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 5,
              save_path: str = None) -> Dict[str, Any]:
        """训练模型"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)
        
        best_val_acc = 0
        patience_counter = 0
        training_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        
        print(f"🚀 开始训练 {self.model.__class__.__name__}...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # 记录历史
            training_history['train_acc'].append(train_metrics['accuracy'])
            training_history['val_acc'].append(val_metrics['accuracy'])
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            
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
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, criterion) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def _validate_epoch(self, val_loader: DataLoader, criterion) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
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
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """评估模型"""
        print(f"📊 评估模型...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
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
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time_ms': avg_inference_time
        }
        
        print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  平均推理时间: {avg_inference_time:.2f}ms")
        
        return results