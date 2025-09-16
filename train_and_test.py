#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的DGA检测模型训练和测试流程
支持所有模型的训练、测试和性能对比
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
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, mode='max'):
        """
        Args:
            patience: 等待改善的轮数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
            mode: 'max' 表示指标越大越好(如准确率), 'min' 表示指标越小越好(如损失)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
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
    
    def __call__(self, score, model):
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
                print(f"早停触发，恢复最佳权重 (最佳分数: {self.best_score:.4f})")
            else:
                print(f"早停触发 (最佳分数: {self.best_score:.4f})")
        
        return self.early_stop

# 导入项目模块
from core.dataset import create_data_loaders
# 只导入基础模型，避免依赖问题
try:
    from models.implementations.cnn_model import CNNModel
except ImportError:
    CNNModel = None

try:
    from models.implementations.bilstm_attention_model import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from models.implementations.simple_lstm_model import SimpleLSTMModel
except ImportError:
    SimpleLSTMModel = None

try:
    from models.implementations.tcbam_models import TCBAMModel
except ImportError:
    TCBAMModel = None
from config.config import config


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        os.makedirs('./data/models', exist_ok=True)
        os.makedirs('./data/results', exist_ok=True)
        os.makedirs('./data/training_logs', exist_ok=True)
    
    def create_model(self, model_name: str, dataset_info: Dict[str, Any]):
        """创建模型实例"""
        # 只包含可用的基础模型
        model_classes = {}
        if CNNModel is not None:
            model_classes['cnn'] = CNNModel
        if LSTMModel is not None:
            model_classes['bilstm_attention'] = LSTMModel
        if SimpleLSTMModel is not None:
            model_classes['simple_lstm'] = SimpleLSTMModel
        if TCBAMModel is not None:
            model_classes['tcbam'] = TCBAMModel
        
        if model_name not in model_classes:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_class = model_classes[model_name]
        
        # 根据模型类型创建实例
        if model_name == 'cnn':
            model = model_class(
                vocab_size=dataset_info['vocab_size'],
                d_model=config.model.d_model,
                num_classes=dataset_info['num_classes']
            )
        elif model_name in ['bilstm_attention', 'simple_lstm']:
            model = model_class(
                vocab_size=dataset_info['vocab_size'],
                d_model=config.model.d_model,
                hidden_size=config.model.d_model,
                num_classes=dataset_info['num_classes']
            )
        elif model_name in ['mamba', 'mambaformer']:
            model = model_class(
                vocab_size=dataset_info['vocab_size'],
                d_model=config.model.d_model,
                num_classes=dataset_info['num_classes']
            )
        elif model_name in ['moe', 'simplified_moe', 'homogeneous_moe']:
            model = model_class(
                vocab_size=dataset_info['vocab_size'],
                d_model=config.model.d_model,
                num_classes=dataset_info['num_classes']
            )
        elif model_name == 'tcbam':
            model = model_class(
                vocab_size=dataset_info['vocab_size'],
                num_classes=dataset_info['num_classes'],
                embed_dim=config.model.d_model,
                hidden_dim=config.model.d_model
            )
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, model_name: str, 
                   num_epochs: int = None) -> Dict[str, Any]:
        """训练模型"""
        if num_epochs is None:
            num_epochs = config.training.num_epochs
        
        print(f"\n开始训练 {model_name} 模型...")
        print(f"训练参数: epochs={num_epochs}, batch_size={config.training.batch_size}, lr={config.training.learning_rate}")
        
        # 优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=config.training.patience//2, factor=0.5
        )
        
        # 早停机制
        early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=0.001,
            restore_best_weights=True,
            mode='max'  # 监控验证准确率，越大越好
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': [],
            'early_stopped': False,
            'stopped_epoch': None
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                if model_name == 'simplified_moe':
                    # SimplifiedImprovedMoEModel需要特殊处理
                    logits, gate_weights, expert_outputs = model(data, return_expert_outputs=True)
                    # 使用模型的compute_total_loss方法
                    loss, loss_components = model.compute_total_loss(logits, target, gate_weights, expert_outputs)
                elif model_name in ['moe', 'homogeneous_moe']:
                    # 其他MoE模型可能返回额外的损失
                    output = model(data)
                    if isinstance(output, tuple):
                        if len(output) == 2:
                            logits, aux_loss = output
                            loss = criterion(logits, target) + aux_loss
                        else:
                            logits = output[0]
                            loss = criterion(logits, target)
                    else:
                        logits = output
                        loss = criterion(logits, target)
                else:
                    logits = model(data)
                    loss = criterion(logits, target)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                pred = logits.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if model_name == 'simplified_moe':
                        logits = model(data)  # 验证时只需要logits
                        loss = criterion(logits, target)
                    else:
                        output = model(data)
                        if isinstance(output, tuple):
                            logits, _ = output
                        else:
                            logits = output
                        loss = criterion(logits, target)
                    val_loss += loss.item()
                    
                    pred = logits.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # 计算平均指标
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['epoch_times'].append(epoch_time)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 打印结果
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # 保存模型
                model_path = f'./data/models/best_{model_name}_model.pth'
                torch.save(model.state_dict(), model_path)
                print(f'  ✓ 保存最佳模型: {model_path} (Val Acc: {val_acc:.2f}%)')
            
            # 早停检查
            if early_stopping(val_acc, model):
                print(f'  早停触发 (patience={config.training.patience})')
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch + 1
                break
            
            print('-' * 60)
        
        # 训练完成
        total_time = sum(history['epoch_times'])
        print(f'\n{model_name} 训练完成!')
        print(f'最佳验证准确率: {best_val_acc:.2f}%')
        print(f'总训练时间: {total_time:.2f}s')
        
        return {
            'model_name': model_name,
            'best_val_acc': best_val_acc,
            'total_time': total_time,
            'history': history,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
    
    def test_model(self, model, test_loader, model_name: str) -> Dict[str, Any]:
        """测试模型"""
        print(f"\n测试 {model_name} 模型...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        inference_times = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Testing {model_name}')
            for data, target in test_pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                if model_name == 'simplified_moe':
                    logits = model(data)  # 测试时只需要logits
                else:
                    output = model(data)
                    if isinstance(output, tuple):
                        logits, _ = output
                    else:
                        logits = output
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                loss = criterion(logits, target)
                total_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        avg_loss = total_loss / len(test_loader)
        avg_inference_time = np.mean(inference_times)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy * 100,
            'f1_score': f1,
            'avg_loss': avg_loss,
            'avg_inference_time': avg_inference_time * 1000,  # ms
            'total_samples': len(all_labels),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'classification_report': classification_report(all_labels, all_predictions)
        }
        
        print(f'{model_name} 测试结果:')
        print(f'  准确率: {results["accuracy"]:.2f}%')
        print(f'  F1分数: {results["f1_score"]:.4f}')
        print(f'  平均损失: {results["avg_loss"]:.4f}')
        print(f'  平均推理时间: {results["avg_inference_time"]:.2f}ms')
        
        return results


def plot_training_history(histories: List[Dict], save_path: str = './data/training_logs'):
    """绘制训练历史"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for history in histories:
        model_name = history['model_name']
        hist = history['history']
        
        # 训练损失
        axes[0, 0].plot(hist['train_loss'], label=f'{model_name} Train')
        axes[0, 0].plot(hist['val_loss'], label=f'{model_name} Val', linestyle='--')
        
        # 训练准确率
        axes[0, 1].plot(hist['train_acc'], label=f'{model_name} Train')
        axes[0, 1].plot(hist['val_acc'], label=f'{model_name} Val', linestyle='--')
        
        # Epoch时间
        axes[1, 0].plot(hist['epoch_times'], label=model_name)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Epoch Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 模型对比
    model_names = [h['model_name'] for h in histories]
    best_accs = [h['best_val_acc'] for h in histories]
    
    axes[1, 1].bar(model_names, best_accs)
    axes[1, 1].set_title('Best Validation Accuracy')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练历史图表已保存到: {save_path}/training_comparison.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DGA检测模型训练和测试')
    parser.add_argument('--models', nargs='+', 
                       choices=['cnn', 'bilstm_attention', 'simple_lstm', 'tcbam', 'all'],
                       default=['all'], help='要训练的模型')
    parser.add_argument('--dataset', type=str, 
                       default='./data/processed/small_dga_dataset.pkl',
                       help='数据集路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (默认使用配置文件)')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备')
    parser.add_argument('--test-only', action='store_true',
                       help='仅测试已训练的模型')
    
    args = parser.parse_args()
    
    # 确定要处理的模型
    if 'all' in args.models:
        models_to_process = ['cnn', 'bilstm_attention', 'simple_lstm', 'tcbam']
    else:
        models_to_process = args.models
    
    print("=" * 80)
    print("DGA检测模型训练和测试流程")
    print("=" * 80)
    print(f"处理模型: {', '.join(models_to_process)}")
    print(f"数据集: {args.dataset}")
    if not args.test_only:
        print(f"训练轮数: {args.epochs or config.training.num_epochs}")
    print(f"仅测试模式: {args.test_only}")
    
    # 加载数据
    try:
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=args.dataset,
            batch_size=config.training.batch_size
        )
        print(f"\n数据集信息:")
        print(f"  训练样本: {dataset_info['train_samples']}")
        print(f"  验证样本: {dataset_info['val_samples']}")
        print(f"  测试样本: {dataset_info['test_samples']}")
        print(f"  类别数: {dataset_info['num_classes']}")
        print(f"  词汇表大小: {dataset_info['vocab_size']}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 创建训练器
    trainer = ModelTrainer(device=args.device)
    
    training_histories = []
    test_results = []
    
    # 处理每个模型
    for model_name in models_to_process:
        print(f"\n{'='*60}")
        print(f"处理模型: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            model = trainer.create_model(model_name, dataset_info)
            
            if not args.test_only:
                # 训练模型
                train_history = trainer.train_model(
                    model, train_loader, val_loader, model_name, args.epochs
                )
                training_histories.append(train_history)
                
                # 重新加载最佳模型进行测试
                model_path = f'./data/models/best_{model_name}_model.pth'
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=trainer.device))
            else:
                # 仅测试模式：加载已训练的模型
                model_path = f'./data/models/best_{model_name}_model.pth'
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=trainer.device))
                    print(f"加载已训练模型: {model_path}")
                else:
                    print(f"警告: 未找到训练好的模型 {model_path}，跳过测试")
                    continue
            
            # 测试模型
            test_result = trainer.test_model(model, test_loader, model_name)
            test_results.append(test_result)
            
        except Exception as e:
            import traceback
            print(f"处理模型 {model_name} 时出错: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            continue
    
    # 保存结果
    if training_histories:
        with open('./data/training_logs/training_histories.pkl', 'wb') as f:
            pickle.dump(training_histories, f)
        plot_training_history(training_histories)
    
    if test_results:
        with open('./data/results/test_results.pkl', 'wb') as f:
            pickle.dump(test_results, f)
        
        # 生成对比报告
        print("\n" + "=" * 80)
        print("模型性能对比报告")
        print("=" * 80)
        print(f"{'模型':<20} {'准确率':<10} {'F1分数':<10} {'推理时间(ms)':<15}")
        print("-" * 65)
        
        for result in test_results:
            print(f"{result['model_name']:<20} "
                  f"{result['accuracy']:<10.2f} "
                  f"{result['f1_score']:<10.4f} "
                  f"{result['avg_inference_time']:<15.2f}")
        
        # 推荐最佳模型
        best_acc_model = max(test_results, key=lambda x: x['accuracy'])
        best_f1_model = max(test_results, key=lambda x: x['f1_score'])
        fastest_model = min(test_results, key=lambda x: x['avg_inference_time'])
        
        print("\n推荐模型:")
        print(f"  最高准确率: {best_acc_model['model_name']} ({best_acc_model['accuracy']:.2f}%)")
        print(f"  最高F1分数: {best_f1_model['model_name']} ({best_f1_model['f1_score']:.4f})")
        print(f"  最快推理: {fastest_model['model_name']} ({fastest_model['avg_inference_time']:.2f}ms)")
    
    print("\n" + "=" * 80)
    print("训练和测试流程完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()