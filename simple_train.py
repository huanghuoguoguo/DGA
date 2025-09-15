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
import numpy as np
from sklearn.model_selection import train_test_split
from core.base_model import ModelTrainer
from core.improved_trainer import ImprovedModelTrainer
from core.enhanced_trainer import EnhancedModelTrainer
from core.enhanced_moe_trainer import EnhancedMoETrainer
from config.config import config
from models.implementations.cnn_model import CNNModel
from models.implementations.lstm_model import LSTMModel
from models.implementations.mamba_model import MambaModel
from models.implementations.moe_model import MoEModel
from models.implementations.improved_moe_model import ImprovedMoEModel
from models.implementations.simplified_improved_moe_model import SimplifiedImprovedMoEModel
from models.implementations.specialized_experts import (
    CharacterLevelExpert, DictionaryLevelExpert, BiGRUAttentionExpert, 
    CNNWithCBAMExpert, TransformerExpert
)
from models.implementations.specialized_moe_model import SpecializedMoEModel
from models.implementations.enhanced_mambaformer_model import EnhancedMambaFormerModel
from models.implementations.advanced_mambaformer_moe import AdvancedMambaFormerMoE
from models.implementations.homogeneous_mambaformer_moe import HomogeneousMambaFormerMoE
from models.implementations.tcbam_models import TCBAMModel
from models.implementations.homogeneous_tcbam_moe import HomogeneousTCBAMMoE
from models.implementations.enhanced_homogeneous_tcbam_moe import EnhancedHomogeneousTCBAMMoE
from models.implementations.mambaformer_model import MambaFormerModel


def get_model(model_name: str, vocab_size: int, num_classes: int = 2):
    """根据名称创建模型"""
    models = {
        'cnn': CNNModel,
        'lstm': LSTMModel,
        'mamba': MambaModel,
        'moe': MoEModel,
        'improved_moe': ImprovedMoEModel,
        'simplified_moe': SimplifiedImprovedMoEModel,
        'mambaformer': MambaFormerModel,
        # 专门化专家模型
        'char_expert': CharacterLevelExpert,
        'dict_expert': DictionaryLevelExpert,
        'bigru_att': BiGRUAttentionExpert,
        'cnn_cbam': CNNWithCBAMExpert,
        'transformer_expert': TransformerExpert,
        # 专门化MoE模型
        'specialized_moe_char_dict': lambda vocab_size, num_classes: SpecializedMoEModel(vocab_size, expert_config="char_dict", num_classes=num_classes),
        'specialized_moe_advanced': lambda vocab_size, num_classes: SpecializedMoEModel(vocab_size, expert_config="advanced", num_classes=num_classes),
        'specialized_moe_hybrid': lambda vocab_size, num_classes: SpecializedMoEModel(vocab_size, expert_config="hybrid", num_classes=num_classes),
        # 增强MambaFormer模型
        'enhanced_mambaformer': EnhancedMambaFormerModel,
        # TCBAM模型
        'tcbam': TCBAMModel,
        # 同构TCBAM-MoE模型
        'homogeneous_tcbam_moe': lambda vocab_size, num_classes: HomogeneousTCBAMMoE(vocab_size, num_classes=num_classes, num_experts=4),
        # 增强版同构TCBAM-MoE模型
        'enhanced_tcbam_moe': lambda vocab_size, num_classes: EnhancedHomogeneousTCBAMMoE(vocab_size, num_classes=num_classes, num_experts=4),
        # 高级MambaFormer MoE模型
        'advanced_moe_mambaformer': lambda vocab_size, num_classes: AdvancedMambaFormerMoE(vocab_size, expert_config="mambaformer_only", num_classes=num_classes),
        'advanced_moe_hybrid': lambda vocab_size, num_classes: AdvancedMambaFormerMoE(vocab_size, expert_config="hybrid_advanced", num_classes=num_classes),
        'advanced_moe_ultimate': lambda vocab_size, num_classes: AdvancedMambaFormerMoE(vocab_size, expert_config="advanced", num_classes=num_classes),
        # 同构MambaFormer MoE模型
        'homogeneous_moe_4experts': lambda vocab_size, num_classes: HomogeneousMambaFormerMoE(vocab_size, num_experts=4, num_classes=num_classes),
        'homogeneous_moe_6experts': lambda vocab_size, num_classes: HomogeneousMambaFormerMoE(vocab_size, num_experts=6, num_classes=num_classes),
        'homogeneous_moe_8experts': lambda vocab_size, num_classes: HomogeneousMambaFormerMoE(vocab_size, num_experts=8, num_classes=num_classes)
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    # 处理lambda函数的特殊情况
    if callable(model_class) and not isinstance(model_class, type):
        # 对于lambda函数，直接调用
        return model_class(vocab_size, num_classes)
    
    if model_name.lower() == 'mambaformer':
        # MambaFormer使用特殊参数
        return model_class(
            vocab_size=vocab_size, 
            d_model=config.model.d_model, 
            n_layers=3,
            d_state=16,
            n_heads=4,
            num_classes=num_classes, 
            dropout=config.model.dropout,
            fusion_type='gated'  # 默认使用门控融合
        )
    elif model_name.lower() == 'tcbam':
        # TCBAM使用特殊参数
        return model_class(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=config.model.d_model,
            hidden_dim=128,
            num_filters=128,
            num_heads=8,
            num_layers=2,
            dropout=config.model.dropout
        )
    else:
        return model_class(
            vocab_size=vocab_size, 
            d_model=config.model.d_model, 
            num_classes=num_classes, 
            dropout=config.model.dropout
        )


def train_model(model_name: str, quick_test: bool = False, dataset_size: str = 'small'):
    """训练指定模型"""
    print(f"🚀 开始训练 {model_name.upper()} 模型")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 选择数据集路径
    dataset_paths = {
        'small': config.data.dataset_path,
        'medium': './data/processed/medium_dga_dataset.pkl',
        'large': './data/processed/large_dga_dataset.pkl',
        'xlarge': './data/processed/xlarge_dga_dataset.pkl',
        'small_multiclass': './data/processed/small_multiclass_dga_dataset.pkl',
        'medium_multiclass': './data/processed/medium_multiclass_dga_dataset.pkl',
        'large_multiclass': './data/processed/large_multiclass_dga_dataset.pkl'
    }
    
    dataset_path = dataset_paths.get(dataset_size, config.data.dataset_path)
    print(f"📂 加载{dataset_size}数据集: {dataset_path}")
    
    try:
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=dataset_path,
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
    print(f"\n🏗️  创建 {model_name.upper()} 模型...")
    
    # 检查是否为多分类数据集
    num_classes = dataset_info.get('num_classes', 2)
    if 'multiclass' in dataset_size:
        print(f"  检测到多分类数据集，类别数: {num_classes}")
    
    model = get_model(model_name, dataset_info['vocab_size'], num_classes)
    model.print_model_info()
    
    # 创建训练器
    if model_name == 'improved_moe':
        trainer = ImprovedModelTrainer(model, device, load_balance_weight=0.1)
        print(f"  使用改进训练器，负载均衡权重: 0.1")
    elif model_name == 'enhanced_tcbam_moe':
        trainer = EnhancedMoETrainer(model, device, load_balance_weight=0.01, diversity_weight=0.01)
        print(f"  使用增强MoE训练器，负载均衡权重: 0.01, 多样性权重: 0.01")
    elif (model_name == 'simplified_moe' or model_name.startswith('specialized_moe') or 
          model_name.startswith('advanced_moe') or model_name.startswith('homogeneous_moe')):
        trainer = EnhancedModelTrainer(model, device)
        print(f"  使用增强训练器，支持多样性损失")
    else:
        trainer = ModelTrainer(model, device)
    
    # 训练参数 - 为MoE模型增加更多训练轮数
    if 'moe' in model_name.lower():
        epochs = 10 if quick_test else 50  # MoE模型需要更多训练
    else:
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
        
        # 专家使用分析（仅对MoE模型）
        expert_analysis = None
        if model_name == 'improved_moe' and hasattr(trainer, 'analyze_expert_usage'):
            print(f"\\n🔍 分析专家使用情况...")
            expert_analysis = trainer.analyze_expert_usage(test_loader)
        elif (model_name == 'simplified_moe' or model_name.startswith('specialized_moe') or 
              model_name.startswith('advanced_moe') or model_name.startswith('homogeneous_moe')) and hasattr(trainer, 'analyze_expert_usage_detailed'):
            print(f"\\n🔍 详细分析专家使用情况...")
            expert_analysis = trainer.analyze_expert_usage_detailed(test_loader)
        
        # 保存结果
        results = {
            'model_name': model_name,
            'model_info': model.get_model_info(),
            'training_results': training_results,
            'test_results': test_results,
            'expert_analysis': expert_analysis
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
                       choices=[
                             'cnn', 'lstm', 'mamba', 'moe', 'improved_moe', 'simplified_moe', 'mambaformer',
                             'char_expert', 'dict_expert', 'bigru_att', 'cnn_cbam', 'transformer_expert',
                             'specialized_moe_char_dict', 'specialized_moe_advanced', 'specialized_moe_hybrid',
                             'enhanced_mambaformer', 'tcbam', 'homogeneous_tcbam_moe', 'enhanced_tcbam_moe',
                             'advanced_moe_mambaformer', 'advanced_moe_hybrid', 'advanced_moe_ultimate',
                              'homogeneous_moe_4experts', 'homogeneous_moe_6experts', 'homogeneous_moe_8experts'
                         ],
                       help='要训练的模型类型')
    parser.add_argument('--quick', action='store_true', 
                       help='快速测试模式（5个epoch）')
    parser.add_argument('--all', action='store_true', 
                       help='训练所有模型')
    parser.add_argument('--dataset', type=str, default='small', 
                        choices=['small', 'medium', 'large', 'xlarge', 'small_multiclass', 'medium_multiclass', 'large_multiclass'],
                        help='数据集大小')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(config.training.random_seed)
    
    if args.all:
        print("🎯 训练所有模型...")
        models = ['cnn', 'lstm', 'mamba', 'moe', 'mambaformer']
        results = {}
        
        for model_name in models:
            print(f"\\n{'='*80}")
            result = train_model(model_name, args.quick, args.dataset)
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
        train_model(args.model, args.quick, args.dataset)


if __name__ == "__main__":
    main()