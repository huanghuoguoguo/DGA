#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练框架
基于配置文件的热插拔模型训练系统
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import time
import argparse
import logging
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score

# 导入核心模块
from core.dataset import create_data_loaders
from core.config_manager import ConfigManager
from core.model_loader import ModelLoader


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 mode: str = 'max', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """检查是否应该早停"""
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logging.info(f"早停触发，恢复最佳权重 (最佳分数: {self.best_score:.4f})")
            else:
                logging.info(f"早停触发 (最佳分数: {self.best_score:.4f})")
        
        return self.early_stop


class TrainingFramework:
    """统一训练框架"""
    
    def __init__(self, config_path: str):
        # 初始化配置管理器
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        
        if self.config is None:
            raise ValueError(f"无法加载配置文件: {config_path}")
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型加载器
        self.model_loader = ModelLoader()
        
        # 设置设备
        self.device = self._setup_device()
        
        # 设置随机种子
        self._setup_seed()
        
        # 创建输出目录
        self._create_output_dirs()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("训练框架初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config_manager.get_logging_config()
        if not log_config:
            return
        
        # 创建日志目录
        log_file = log_config.get('file')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file) if log_file else logging.NullHandler(),
                logging.StreamHandler() if log_config.get('console', True) else logging.NullHandler()
            ]
        )
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config_manager.get_device_config()
        device_type = device_config.get('type', 'auto') if device_config else 'auto'
        
        if device_type == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_type)
        
        logging.info(f"使用设备: {device}")
        return device
    
    def _setup_seed(self):
        """设置随机种子"""
        exp_config = self.config_manager.get_experiment_config()
        if not exp_config:
            return
        
        seed = exp_config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if exp_config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif exp_config.get('benchmark', False):
            torch.backends.cudnn.benchmark = True
        
        logging.info(f"设置随机种子: {seed}")
    
    def _create_output_dirs(self):
        """创建输出目录"""
        output_config = self.config_manager.get_output_config()
        if not output_config:
            return
        
        dirs = ['model_save_dir', 'results_save_dir', 'checkpoint_dir']
        for dir_key in dirs:
            if dir_key in output_config:
                os.makedirs(output_config[dir_key], exist_ok=True)
    
    def load_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
        """加载数据集"""
        dataset_config = self.config_manager.get_dataset_config()
        
        self.logger.info(f"加载数据集: {dataset_config['type']}")
        
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=dataset_config['path'],
            batch_size=dataset_config['batch_size']
        )
        
        self.logger.info(f"数据集加载完成:")
        self.logger.info(f"  训练集: {dataset_info['train_samples']:,} 样本")
        self.logger.info(f"  验证集: {dataset_info['val_samples']:,} 样本")
        self.logger.info(f"  测试集: {dataset_info['test_samples']:,} 样本")
        self.logger.info(f"  类别数: {dataset_info['num_classes']}")
        self.logger.info(f"  词汇表大小: {dataset_info['vocab_size']}")
        
        return train_loader, val_loader, test_loader, dataset_info
    
    def create_model(self, model_name: str, dataset_info: Dict[str, Any]) -> nn.Module:
        """创建模型"""
        models_config = self.config_manager.get_models_config()
        
        if model_name not in models_config:
            raise ValueError(f"模型 {model_name} 未在配置中找到")
        
        model_config = models_config[model_name]
        
        if not model_config.get('enabled', True):
            raise ValueError(f"模型 {model_name} 未启用")
        
        model = self.model_loader.create_model(model_config, dataset_info)
        
        if model is None:
            raise ValueError(f"无法创建模型 {model_name}")
        
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型 {model_name} 创建成功:")
        self.logger.info(f"  总参数量: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器"""
        training_config = self.config_manager.get_training_config()
        
        optimizer_type = training_config.get('optimizer', 'adam').lower()
        lr = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        self.logger.info(f"创建优化器: {optimizer_type}, lr={lr}, weight_decay={weight_decay}")
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer):
        """创建学习率调度器"""
        training_config = self.config_manager.get_training_config()
        
        scheduler_type = training_config.get('scheduler', 'plateau').lower()
        
        if scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )
        elif scheduler_type == 'step':
            step_size = training_config.get('step_size', 10)
            gamma = training_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = training_config['epochs']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:
            scheduler = None
        
        if scheduler:
            self.logger.info(f"创建学习率调度器: {scheduler_type}")
        
        return scheduler
    
    def create_early_stopping(self) -> Optional[EarlyStopping]:
        """创建早停机制"""
        training_config = self.config_manager.get_training_config()
        
        if not training_config.get('early_stopping', False):
            return None
        
        early_stopping = EarlyStopping(
            patience=training_config.get('patience', 5),
            min_delta=training_config.get('min_delta', 0.001),
            mode=training_config.get('mode', 'max'),
            restore_best_weights=True
        )
        
        self.logger.info(f"启用早停机制: patience={early_stopping.patience}")
        return early_stopping
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> Dict[str, Any]:
        """训练模型"""
        training_config = self.config_manager.get_training_config()
        
        # 创建优化器和调度器
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        early_stopping = self.create_early_stopping()
        
        # 训练参数
        num_epochs = training_config['epochs']
        grad_clip = training_config.get('grad_clip', 1.0)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': []
        }
        
        best_val_accuracy = 0.0
        start_time = time.time()
        
        self.logger.info(f"开始训练模型 {model_name}")
        self.logger.info(f"训练参数: epochs={num_epochs}, grad_clip={grad_clip}")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # 更新进度条
                current_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            # 验证阶段
            val_results = self.evaluate_model(model, val_loader)
            
            # 计算平均指标
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_accuracy = 100. * train_correct / train_total
            epoch_val_loss = val_results['loss']
            epoch_val_accuracy = val_results['accuracy'] * 100
            
            # 记录历史
            history['train_losses'].append(epoch_train_loss)
            history['train_accuracies'].append(epoch_train_accuracy)
            history['val_losses'].append(epoch_val_loss)
            history['val_accuracies'].append(epoch_val_accuracy)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_accuracy)
                else:
                    scheduler.step()
            
            # 打印结果
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Acc: {epoch_train_accuracy:.2f}%, "
                f"Val Acc: {epoch_val_accuracy:.2f}%, "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Val Loss: {epoch_val_loss:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # 保存最佳模型
            if epoch_val_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_val_accuracy
                self._save_model(model, model_name, 'best')
                self.logger.info(f"保存最佳模型: {best_val_accuracy:.2f}%")
            
            # 早停检查
            if early_stopping and early_stopping(epoch_val_accuracy, model):
                self.logger.info(f"早停触发，停止训练")
                break
        
        training_time = time.time() - start_time
        
        # 保存最终模型
        output_config = self.config_manager.get_output_config()
        if output_config and not output_config.get('save_best_only', True):
            self._save_model(model, model_name, 'last')
        
        self.logger.info(f"模型 {model_name} 训练完成")
        self.logger.info(f"最佳验证准确率: {best_val_accuracy:.2f}%")
        self.logger.info(f"训练时间: {training_time:.2f}秒")
        
        return {
            'model_name': model_name,
            'best_val_accuracy': best_val_accuracy,
            'training_time': training_time,
            'history': history
        }
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        inference_times = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                output = model(data)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        avg_loss = total_loss / len(data_loader)
        avg_inference_time = np.mean(inference_times)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'loss': avg_loss,
            'avg_inference_time': avg_inference_time
        }
    
    def _save_model(self, model: nn.Module, model_name: str, suffix: str = 'best'):
        """保存模型"""
        output_config = self.config_manager.get_output_config()
        if not output_config:
            return
        
        save_dir = output_config.get('model_save_dir', './results/models')
        model_path = os.path.join(save_dir, f"{model_name}_{suffix}.pth")
        
        torch.save(model.state_dict(), model_path)
    
    def save_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        output_config = self.config_manager.get_output_config()
        if not output_config:
            return
        
        results_dir = output_config.get('results_save_dir', './results/experiments')
        exp_config = self.config_manager.get_experiment_config()
        exp_name = exp_config.get('name', 'experiment') if exp_config else 'experiment'
        
        results_path = os.path.join(results_dir, f"{exp_name}_results.pkl")
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"结果已保存到: {results_path}")
    
    def run_training(self, model_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """运行训练流程"""
        # 加载数据集
        train_loader, val_loader, test_loader, dataset_info = self.load_dataset()
        
        # 获取要训练的模型
        if model_names is None:
            enabled_models = self.config_manager.get_enabled_models()
            model_names = list(enabled_models.keys())
        
        if not model_names:
            raise ValueError("没有指定要训练的模型")
        
        self.logger.info(f"将训练以下模型: {model_names}")
        
        all_results = []
        
        for model_name in model_names:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"开始训练模型: {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                # 创建模型
                model = self.create_model(model_name, dataset_info)
                
                # 训练模型
                training_results = self.train_model(model, train_loader, val_loader, model_name)
                
                # 测试模型
                test_results = self.evaluate_model(model, test_loader)
                
                # 合并结果
                combined_results = {
                    **training_results,
                    'test_accuracy': test_results['accuracy'] * 100,
                    'test_f1_score': test_results['f1_score'],
                    'test_loss': test_results['loss'],
                    'avg_inference_time': test_results['avg_inference_time']
                }
                
                all_results.append(combined_results)
                
                self.logger.info(f"模型 {model_name} 测试结果:")
                self.logger.info(f"  测试准确率: {test_results['accuracy']*100:.2f}%")
                self.logger.info(f"  测试F1分数: {test_results['f1_score']:.4f}")
                self.logger.info(f"  平均推理时间: {test_results['avg_inference_time']:.2f}ms")
                
            except Exception as e:
                self.logger.error(f"训练模型 {model_name} 时发生错误: {e}")
                continue
        
        # 保存所有结果
        if all_results:
            self.save_results(all_results)
            
            # 打印总结
            self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """打印训练总结"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("训练完成！性能总结:")
        self.logger.info(f"{'='*80}")
        
        header = f"{'模型':<15} {'验证准确率':<12} {'测试准确率':<12} {'训练时间(s)':<12} {'推理时间(ms)':<12}"
        self.logger.info(header)
        self.logger.info("-" * 70)
        
        for result in results:
            line = (f"{result['model_name']:<15} "
                   f"{result['best_val_accuracy']:<11.2f}% "
                   f"{result['test_accuracy']:<11.2f}% "
                   f"{result['training_time']:<11.1f} "
                   f"{result['avg_inference_time']:<11.2f}")
            self.logger.info(line)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于配置文件的统一训练框架')
    parser.add_argument('--config', type=str, default='./config/train_config.toml',
                       help='配置文件路径')
    parser.add_argument('--models', nargs='+', default=None,
                       help='要训练的模型名称（默认训练所有启用的模型）')
    
    args = parser.parse_args()
    
    try:
        # 创建训练框架
        framework = TrainingFramework(args.config)
        
        # 运行训练
        results = framework.run_training(args.models)
        
        if not results:
            print("没有成功训练任何模型")
            return 1
        
        print(f"\n成功训练了 {len(results)} 个模型")
        return 0
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())