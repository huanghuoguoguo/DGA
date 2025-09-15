#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 大规模多分类数据集构建器
用于构建大规模多分类的训练和测试数据集
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multiclass_dataset_builder import MultiClassDGADatasetBuilder

def main():
    """构建大规模多分类数据集"""
    print("=== 大规模DGA多分类数据集构建器 ===")
    
    # 创建构建器
    builder = MultiClassDGADatasetBuilder(max_length=40)  # 使用较短的序列长度以提高效率
    
    # 构建大规模多分类数据集
    print("\n构建大规模多分类数据集...")
    X, y, class_names = builder.build_multiclass_dataset(
        dga_families=['conficker', 'zeus', 'necurs', 'locky', 'ramnit', 'cryptolocker', 'dyre', 'matsnu'],
        samples_per_family=2000,  # 每个家族2000个样本
        benign_samples=3000,      # 3000个良性样本
        include_benign=True
    )
    
    print(f"\n数据集统计:")
    print(f"总样本数: {len(X)}")
    print(f"类别数: {len(class_names)}")
    print(f"类别名称: {class_names}")
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = builder.split_dataset(X, y)
    
    # 保存数据集
    builder.save_dataset(
        X_train, X_val, X_test, y_train, y_val, y_test, class_names,
        "./data/processed/large_multiclass_dga_dataset.pkl"
    )
    
    print("\n大规模多分类数据集构建完成！")
    print(f"数据集包含 {len(class_names)} 个类别，共 {len(X)} 个样本")
    print("类别分布:")
    import numpy as np
    for i, class_name in enumerate(class_names):
        count = np.sum(y == i)
        print(f"  {class_name}: {count} 样本")

if __name__ == "__main__":
    main()