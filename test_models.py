#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA检测模型测试脚本
提供完整的模型测试功能，包括准确率、F1分数、混淆矩阵等评估指标
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from typing import Dict, Any, Tuple
from tqdm import tqdm
import time

# 导入项目模块
from core.dataset import create_data_loaders
from models.implementations.cnn_model import CNNModel
from models.implementations.lstm_model import LSTMModel
from models.implementations.mamba_model import MambaModel
from models.implementations.moe_model import MoEModel
from models.implementations.simplified_improved_moe_model import SimplifiedImprovedMoE
from config.config import config


class ModelTester:
    """模型测试器"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 创建结果保存目录
        os.makedirs('./data/test_results', exist_ok=True)
    
    def load_model(self, model_class, model_path: str, dataset_info: Dict[str, Any]):
        """加载训练好的模型"""
        try:
            # 初始化模型
            if model_class == CNNModel:
                model = model_class(
                    vocab_size=dataset_info['vocab_size'],
                    embed_dim=config.model.d_model,
                    num_classes=dataset_info['num_classes']
                )
            elif model_class == LSTMModel:
                model = model_class(
                    vocab_size=dataset_info['vocab_size'],
                    embed_dim=config.model.d_model,
                    hidden_dim=config.model.d_model,
                    num_classes=dataset_info['num_classes']
                )
            elif model_class == MambaModel:
                model = model_class(
                    vocab_size=dataset_info['vocab_size'],
                    d_model=config.model.d_model,
                    num_classes=dataset_info['num_classes']
                )
            elif model_class in [MoEModel, SimplifiedImprovedMoE]:
                model = model_class(
                    vocab_size=dataset_info['vocab_size'],
                    d_model=config.model.d_model,
                    num_classes=dataset_info['num_classes']
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_class}")
            
            # 加载权重
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"成功加载模型: {model_path}")
            else:
                print(f"警告: 模型文件不存在 {model_path}，将使用随机初始化的模型")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    def evaluate_model(self, model, test_loader, model_name: str) -> Dict[str, Any]:
        """评估模型性能"""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        inference_times = []
        
        criterion = nn.CrossEntropyLoss()
        
        print(f"\n评估模型: {model_name}")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="测试进度")):
                data, target = data.to(self.device), target.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 计算损失
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # 获取预测结果
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        avg_loss = total_loss / len(test_loader)
        avg_inference_time = np.mean(inference_times)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy * 100,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'avg_loss': avg_loss,
            'avg_inference_time': avg_inference_time * 1000,  # 转换为毫秒
            'confusion_matrix': cm,
            'classification_report': classification_report(all_labels, all_predictions),
            'total_samples': len(all_labels)
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, model_name: str, class_names=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 保存图片
        plt.savefig(f'./data/test_results/{model_name}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_all_models(self, dataset_path: str = './data/processed/small_dga_dataset.pkl'):
        """测试所有可用的模型"""
        print("=" * 60)
        print("DGA检测模型测试")
        print("=" * 60)
        
        # 加载数据
        try:
            _, _, test_loader, dataset_info = create_data_loaders(
                dataset_path=dataset_path,
                batch_size=config.training.batch_size
            )
            print(f"数据集信息:")
            print(f"  - 测试样本数: {dataset_info['test_samples']}")
            print(f"  - 类别数: {dataset_info['num_classes']}")
            print(f"  - 词汇表大小: {dataset_info['vocab_size']}")
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return
        
        # 定义要测试的模型
        models_to_test = [
            (CNNModel, 'CNN', 'best_cnn_model.pth'),
            (LSTMModel, 'LSTM', 'best_lstm_model.pth'),
            (MambaModel, 'Mamba', 'best_mamba_model.pth'),
            (MoEModel, 'MoE', 'best_moe_model.pth'),
            (SimplifiedImprovedMoE, 'SimplifiedMoE', 'best_simplifiedmoe_model.pth')
        ]
        
        all_results = []
        
        for model_class, model_name, model_file in models_to_test:
            model_path = os.path.join('./data/models', model_file)
            
            # 加载模型
            model = self.load_model(model_class, model_path, dataset_info)
            if model is None:
                continue
            
            # 评估模型
            results = self.evaluate_model(model, test_loader, model_name)
            all_results.append(results)
            
            # 绘制混淆矩阵
            self.plot_confusion_matrix(
                results['confusion_matrix'], 
                model_name, 
                dataset_info.get('class_names', None)
            )
            
            # 打印结果
            print(f"\n{model_name} 测试结果:")
            print(f"  准确率: {results['accuracy']:.2f}%")
            print(f"  F1分数: {results['f1_score']:.4f}")
            print(f"  精确率: {results['precision']:.4f}")
            print(f"  召回率: {results['recall']:.4f}")
            print(f"  平均损失: {results['avg_loss']:.4f}")
            print(f"  平均推理时间: {results['avg_inference_time']:.2f}ms")
        
        # 保存所有结果
        if all_results:
            results_file = './data/test_results/all_models_test_results.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"\n所有测试结果已保存到: {results_file}")
            
            # 生成对比报告
            self.generate_comparison_report(all_results)
    
    def generate_comparison_report(self, all_results):
        """生成模型对比报告"""
        print("\n" + "=" * 60)
        print("模型性能对比报告")
        print("=" * 60)
        
        # 创建对比表格
        print(f"{'模型':<15} {'准确率':<10} {'F1分数':<10} {'推理时间(ms)':<15} {'样本数':<10}")
        print("-" * 60)
        
        best_accuracy = 0
        best_f1 = 0
        fastest_model = None
        fastest_time = float('inf')
        
        for result in all_results:
            print(f"{result['model_name']:<15} "
                  f"{result['accuracy']:<10.2f} "
                  f"{result['f1_score']:<10.4f} "
                  f"{result['avg_inference_time']:<15.2f} "
                  f"{result['total_samples']:<10}")
            
            # 记录最佳性能
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_accuracy_model = result['model_name']
            
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_f1_model = result['model_name']
            
            if result['avg_inference_time'] < fastest_time:
                fastest_time = result['avg_inference_time']
                fastest_model = result['model_name']
        
        print("\n推荐模型:")
        print(f"  最高准确率: {best_accuracy_model} ({best_accuracy:.2f}%)")
        print(f"  最高F1分数: {best_f1_model} ({best_f1:.4f})")
        print(f"  最快推理: {fastest_model} ({fastest_time:.2f}ms)")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DGA检测模型测试')
    parser.add_argument('--dataset', type=str, 
                       default='./data/processed/small_dga_dataset.pkl',
                       help='数据集路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelTester(device=args.device)
    
    # 运行测试
    tester.test_all_models(dataset_path=args.dataset)


if __name__ == '__main__':
    main()