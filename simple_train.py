#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 简单训练脚本
"""

import torch
import pickle
import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.base_model import ModelTrainer
from config.config import config
from models.implementations.cnn_model import CNNModel
from models.implementations.lstm_model import LSTMModel
from models.implementations.mamba_model import MambaModel
from models.implementations.moe_model import MoEModel
from models.implementations.mambaformer_model import MambaFormerModel


def get_model(model_name: str, vocab_size: int):
    """根据名称创建模型"""
    models = {
        'cnn': CNNModel,
        'lstm': LSTMModel,
        'mamba': MambaModel,
        'moe': MoEModel,
        'mambaformer': MambaFormerModel
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    if model_name.lower() == 'mambaformer':
        # MambaFormer使用特殊参数
        return model_class(
            vocab_size=vocab_size, 
            d_model=config.model.d_model, 
            n_layers=3,
            d_state=16,
            n_heads=4,
            num_classes=config.model.num_classes, 
            dropout=config.model.dropout,
            fusion_type='gated'  # 默认使用门控融合
        )
    else:
        return model_class(
            vocab_size=vocab_size, 
            d_model=config.model.d_model, 
            num_classes=config.model.num_classes, 
            dropout=config.model.dropout
        )


def train_model(model_name: str, quick_test: bool = False):
    """训练指定模型"""
    print(f"🚀 开始训练 {model_name.upper()} 模型")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("📂 加载数据集...")
    try:
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=config.data.dataset_path,
            batch_size=config.training.batch_size,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            random_seed=config.training.random_seed
        )
        print_dataset_info(dataset_info)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("请确保数据集文件存在，或先运行数据预处理脚本")
        return None
    
    # 创建模型
    print(f"\\n🏗️  创建 {model_name.upper()} 模型...")
    model = get_model(model_name, dataset_info['vocab_size'])
    model.print_model_info()
    
    # 创建训练器
    trainer = ModelTrainer(model, device)
    
    # 训练参数
    epochs = 5 if quick_test else config.training.num_epochs
    
    # 开始训练
    save_path = config.get_model_save_path(model_name)
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
        save_path=save_path
    )
    
    # 加载最佳模型并评估
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"\\n📊 在测试集上评估最佳模型...")
        test_results = trainer.evaluate(test_loader)
        
        # 保存结果
        results = {
            'model_name': model_name,
            'model_info': model.get_model_info(),
            'training_results': training_results,
            'test_results': test_results
        }
        
        results_path = config.get_results_save_path(model_name)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\\n✅ {model_name.upper()} 训练完成!")
        print(f"最佳验证准确率: {training_results['best_val_accuracy']:.2f}%")
        print(f"测试准确率: {test_results['accuracy']*100:.2f}%")
        print(f"结果已保存到: {results_path}")
        
        return results
    else:
        print(f"❌ 模型文件未找到: {save_path}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DGA检测模型训练')
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'lstm', 'mamba', 'moe', 'mambaformer'],
                       help='要训练的模型类型')
    parser.add_argument('--quick', action='store_true', 
                       help='快速测试模式（5个epoch）')
    parser.add_argument('--all', action='store_true', 
                       help='训练所有模型')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(config.training.random_seed)
    
    if args.all:
        print("🎯 训练所有模型...")
        models = ['cnn', 'lstm', 'mamba', 'moe', 'mambaformer']
        results = {}
        
        for model_name in models:
            print(f"\\n{'='*80}")
            result = train_model(model_name, args.quick)
            if result:
                results[model_name] = result
            print(f"{'='*80}")
        
        print(f"\\n🏆 所有模型训练完成!")
        print(f"训练了 {len(results)} 个模型")
        
        # 简单对比
        if results:
            print(f"\\n📊 性能对比:")
            print(f"{'模型':<10} {'验证准确率':<12} {'测试准确率':<12} {'参数量':<12}")
            print("-" * 50)
            for name, result in results.items():
                val_acc = result['training_results']['best_val_accuracy']
                test_acc = result['test_results']['accuracy'] * 100
                params = result['model_info']['total_params']
                print(f"{name.upper():<10} {val_acc:<11.2f}% {test_acc:<11.2f}% {params:<11,}")
    else:
        train_model(args.model, args.quick)


if __name__ == "__main__":
    main()