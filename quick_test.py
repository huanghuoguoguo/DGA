#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 快速测试脚本
验证重构后的项目是否能正常运行
"""

import torch
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def test_imports():
    """测试所有模块是否能正常导入"""
    print("🔍 测试模块导入...")
    
    try:
        from core.dataset import create_data_loaders, print_dataset_info
        print("  ✅ core.dataset")
    except Exception as e:
        print(f"  ❌ core.dataset: {e}")
        return False
    
    try:
        from core.base_model import BaseModel, ModelTrainer
        print("  ✅ core.base_model")
    except Exception as e:
        print(f"  ❌ core.base_model: {e}")
        return False
    
    try:
        from config.config import config
        print("  ✅ config.config")
    except Exception as e:
        print(f"  ❌ config.config: {e}")
        return False
    
    try:
        from models.implementations.cnn_model import CNNModel
        print("  ✅ models.implementations.cnn_model")
    except Exception as e:
        print(f"  ❌ models.implementations.cnn_model: {e}")
        return False
    
    try:
        from models.implementations.lstm_model import LSTMModel
        print("  ✅ models.implementations.lstm_model")
    except Exception as e:
        print(f"  ❌ models.implementations.lstm_model: {e}")
        return False
    
    try:
        from models.implementations.mamba_model import MambaModel
        print("  ✅ models.implementations.mamba_model")
    except Exception as e:
        print(f"  ❌ models.implementations.mamba_model: {e}")
        return False
    
    return True


def test_data_loading():
    """测试数据加载"""
    print("\\n📂 测试数据加载...")
    
    try:
        from core.dataset import load_dataset
        
        # 尝试加载数据集
        dataset_paths = [
            './data/processed/small_dga_dataset.pkl',
            './data/small_dga_dataset.pkl'
        ]
        
        dataset = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset = load_dataset(path)
                print(f"  ✅ 成功加载数据集: {path}")
                break
        
        if dataset is None:
            print("  ⚠️  未找到数据集文件")
            print("  可用的数据集路径:")
            for path in dataset_paths:
                exists = "存在" if os.path.exists(path) else "不存在"
                print(f"    {path}: {exists}")
            return False
        
        print(f"  📊 数据集信息:")
        print(f"    样本数: {len(dataset['X'])}")
        print(f"    词汇表大小: {dataset['vocab_size']}")
        print(f"    特征维度: {dataset['X'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\\n🏗️  测试模型创建...")
    
    try:
        from models.implementations.cnn_model import CNNModel
        from models.implementations.lstm_model import LSTMModel
        from models.implementations.mamba_model import MambaModel
        
        vocab_size = 40  # 模拟词汇表大小
        
        # 测试CNN模型
        cnn_model = CNNModel(vocab_size=vocab_size, d_model=64)
        print(f"  ✅ CNN模型创建成功, 参数量: {sum(p.numel() for p in cnn_model.parameters()):,}")
        
        # 测试LSTM模型
        lstm_model = LSTMModel(vocab_size=vocab_size, d_model=64)
        print(f"  ✅ LSTM模型创建成功, 参数量: {sum(p.numel() for p in lstm_model.parameters()):,}")
        
        # 测试Mamba模型
        mamba_model = MambaModel(vocab_size=vocab_size, d_model=64, n_layers=2)
        print(f"  ✅ Mamba模型创建成功, 参数量: {sum(p.numel() for p in mamba_model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\\n⚡ 测试模型前向传播...")
    
    try:
        from models.implementations.cnn_model import CNNModel
        
        # 创建模型和虚拟数据
        vocab_size = 40
        seq_length = 20
        batch_size = 4
        
        model = CNNModel(vocab_size=vocab_size, d_model=64)
        
        # 创建虚拟输入
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # 前向传播
        with torch.no_grad():
            output = model(x)
        
        print(f"  ✅ 前向传播成功")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 DGA检测项目重构验证")
    print("=" * 50)
    
    tests = [
        ("模块导入测试", test_imports),
        ("数据加载测试", test_data_loading),
        ("模型创建测试", test_model_creation),
        ("前向传播测试", test_forward_pass)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🧪 {test_name}")
        if test_func():
            passed += 1
        
    print(f"\\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目重构成功！")
        print("\\n📝 接下来可以:")
        print("  1. 运行快速训练测试: python simple_train.py --model cnn --quick")
        print("  2. 训练所有模型: python simple_train.py --all")
        print("  3. 训练特定模型: python simple_train.py --model mamba")
    else:
        print("❌ 部分测试失败，请检查错误信息")
        
    return passed == total


if __name__ == "__main__":
    main()