#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MambaFormer 快速测试
"""

import torch
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.implementations.mambaformer_model import MambaFormerModel

def quick_test():
    print("🚀 MambaFormer快速测试")
    print("=" * 50)
    
    # 测试参数
    vocab_size = 40
    seq_length = 20
    batch_size = 2
    
    print("📋 MambaFormer架构特点:")
    print("  • 结合Mamba状态空间模型和Transformer")
    print("  • 三种融合策略: Sequential, Parallel, Gated")
    print("  • Mamba: 线性复杂度 + 选择性机制")
    print("  • Transformer: 全局注意力 + 并行计算")
    
    print(f"\n🧪 测试不同融合策略:")
    
    for fusion_type in ['sequential', 'parallel', 'gated']:
        try:
            print(f"\n  {fusion_type.upper()}:")
            
            # 创建模型
            model = MambaFormerModel(
                vocab_size=vocab_size,
                d_model=64,
                n_layers=2,
                d_state=16,
                n_heads=4,
                fusion_type=fusion_type
            )
            
            # 模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {total_params:,}")
            
            # 前向传播测试
            x = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                output = model(x)
            
            print(f"    输入: {x.shape} → 输出: {output.shape}")
            print(f"    输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
            print(f"    ✅ 成功")
            
        except Exception as e:
            print(f"    ❌ 失败: {e}")
    
    print(f"\n💡 MambaFormer优势:")
    print("  1. 结合线性复杂度和全局注意力")
    print("  2. 适合长序列处理（域名可能很长）")
    print("  3. 灵活的融合策略选择")
    print("  4. 在DGA检测中平衡效率和准确率")
    
    print(f"\n🎯 推荐用法:")
    print("  python main.py train --model mambaformer --quick")
    print("  python main.py analyze --chart")

if __name__ == "__main__":
    quick_test()