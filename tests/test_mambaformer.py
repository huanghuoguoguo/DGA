#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - MambaFormer模型测试脚本
"""

import torch
import pickle
import sys
import os
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from core.dataset import create_data_loaders, print_dataset_info
from core.base_model import ModelTrainer
from config.config import config
from models.implementations.mambaformer_model import MambaFormerModel


def test_mambaformer_architectures():
    """测试不同的MambaFormer融合策略"""
    print("🚀 MambaFormer架构测试")
    print("=" * 60)
    
    # 设置设备和随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("\n📂 加载数据集...")
    try:
        train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
            dataset_path=config.data.dataset_path,
            batch_size=16,  # 减小batch size以适应更大的模型
            random_seed=42
        )
        print_dataset_info(dataset_info)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 测试不同的融合策略
    fusion_types = ['sequential', 'parallel', 'gated']
    results = {}
    
    for fusion_type in fusion_types:
        print(f"\n{'='*60}")
        print(f"🧪 测试 {fusion_type.upper()} MambaFormer")
        print(f"{'='*60}")
        
        # 创建模型
        model = MambaFormerModel(
            vocab_size=dataset_info['vocab_size'],
            d_model=128,  # 减小模型以加快训练
            n_layers=2,   # 减少层数
            d_state=16,
            n_heads=4,    # 减少注意力头数
            num_classes=2,
            dropout=0.1,
            fusion_type=fusion_type
        )
        
        print(f"📋 {fusion_type.upper()} MambaFormer模型信息:")
        model.print_model_info()
        fusion_info = model.get_fusion_info()
        print(f"  融合策略: {fusion_info['fusion_type']}")
        print(f"  架构类型: {fusion_info['architecture']}")
        
        # 创建训练器
        trainer = ModelTrainer(model, device)
        
        # 快速训练（少数epochs用于测试）
        save_path = f"./data/models/best_mambaformer_{fusion_type}_model.pth"
        
        start_time = time.time()
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,  # 快速测试
            learning_rate=0.001,
            weight_decay=1e-4,
            patience=3,
            save_path=save_path
        )
        training_time = time.time() - start_time
        
        # 测试评估
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            test_results = trainer.evaluate(test_loader)
            
            # 保存结果
            result = {
                'fusion_type': fusion_type,
                'model_info': model.get_model_info(),
                'fusion_info': model.get_fusion_info(),
                'training_results': training_results,
                'test_results': test_results,
                'training_time': training_time
            }
            
            results[fusion_type] = result
            
            # 保存到文件
            result_path = f"./data/results/mambaformer_{fusion_type}_results.pkl"
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            
            print(f"✅ {fusion_type.upper()} MambaFormer测试完成:")
            print(f"  最佳验证准确率: {training_results['best_val_accuracy']:.2f}%")
            print(f"  测试准确率: {test_results['accuracy']*100:.2f}%")
            print(f"  F1分数: {test_results['f1']:.4f}")
            print(f"  推理时间: {test_results['inference_time_ms']:.2f}ms")
            print(f"  训练用时: {training_time:.1f}秒")
            print(f"  结果已保存: {result_path}")
    
    # 对比分析
    if results:
        print(f"\n{'='*80}")
        print("📊 MambaFormer架构对比分析")
        print(f"{'='*80}")
        
        print(f"{'融合策略':<12} {'验证准确率':<12} {'测试准确率':<12} {'F1分数':<10} {'推理时间(ms)':<12} {'参数量':<12}")
        print("-" * 80)
        
        best_fusion = None
        best_accuracy = 0
        
        for fusion_type, result in results.items():
            val_acc = result['training_results']['best_val_accuracy']
            test_acc = result['test_results']['accuracy'] * 100
            f1_score = result['test_results']['f1']
            inference_time = result['test_results']['inference_time_ms']
            params = result['model_info']['total_params']
            
            print(f"{fusion_type.upper():<12} {val_acc:<11.2f}% {test_acc:<11.2f}% "
                  f"{f1_score:<9.4f} {inference_time:<11.2f} {params:<11,}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_fusion = fusion_type
        
        print(f"\n🏆 最佳融合策略: {best_fusion.upper()} ({best_accuracy:.2f}%)")
        
        # 融合策略分析
        print(f"\n💡 融合策略分析:")
        print(f"  Sequential: Mamba处理后再用Transformer，适合顺序特征提取")
        print(f"  Parallel: 并行处理后融合，平衡两种架构的优势")
        print(f"  Gated: 动态权重融合，智能选择最适合的特征")
        
        # 保存对比结果
        comparison_result = {
            'comparison_type': 'MambaFormer_Fusion_Strategies',
            'results': results,
            'best_fusion': best_fusion,
            'best_accuracy': best_accuracy
        }
        
        with open('./data/results/mambaformer_comparison.pkl', 'wb') as f:
            pickle.dump(comparison_result, f)
        
        print(f"\n💾 对比结果已保存: ./data/results/mambaformer_comparison.pkl")
    
    return results


def test_forward_pass():
    """测试MambaFormer前向传播"""
    print("\n🧪 MambaFormer前向传播测试")
    print("-" * 40)
    
    vocab_size = 40
    seq_length = 20
    batch_size = 2
    
    # 测试不同融合策略
    for fusion_type in ['sequential', 'parallel', 'gated']:
        try:
            model = MambaFormerModel(
                vocab_size=vocab_size,
                d_model=64,
                n_layers=2,
                fusion_type=fusion_type
            )
            
            # 创建测试输入
            x = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            # 前向传播
            with torch.no_grad():
                output = model(x)
            
            print(f"✅ {fusion_type.upper()}: 输入{x.shape} → 输出{output.shape}")
            
        except Exception as e:
            print(f"❌ {fusion_type.upper()}: {e}")


def main():
    """主函数"""
    print("🔬 DGA检测 - MambaFormer模型测试")
    print("=" * 80)
    
    # 基础前向传播测试
    test_forward_pass()
    
    # 完整架构测试
    results = test_mambaformer_architectures()
    
    if results:
        print(f"\n🎉 MambaFormer测试完成！")
        print(f"共测试了 {len(results)} 种融合策略")
        print(f"所有结果已保存到 ./data/results/ 目录")
    else:
        print(f"\n❌ 测试未完成，请检查数据集和环境配置")


if __name__ == "__main__":
    main()