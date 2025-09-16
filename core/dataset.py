#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 统一数据集处理模块
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Any


class DGADataset(Dataset):
    """统一的DGA数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(file_path: str = './data/processed/small_dga_dataset.pkl') -> Dict[str, Any]:
    """加载预处理的数据集"""
    if not os.path.exists(file_path):
        # 尝试原始位置
        fallback_path = './data/small_dga_dataset.pkl'
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def create_data_loaders(dataset_path: str = './data/processed/small_dga_dataset.pkl',
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """创建训练、验证、测试数据加载器"""
    
    # 加载数据集
    dataset = load_dataset(dataset_path)
    
    # 检查数据集结构类型
    if 'train' in dataset and 'val' in dataset and 'test' in dataset:
        # 新格式的xlarge_multiclass数据集
        train_data = dataset['train']
        val_data = dataset['val']
        test_data = dataset['test']
        info = dataset['info']
        
        # 转换为numpy数组
        X_train = np.array(train_data['sequences'])
        y_train = np.array(train_data['labels'])
        X_val = np.array(val_data['sequences'])
        y_val = np.array(val_data['labels'])
        X_test = np.array(test_data['sequences'])
        y_test = np.array(test_data['labels'])
        
        # 创建数据集对象
        train_dataset = DGADataset(X_train, y_train)
        val_dataset = DGADataset(X_val, y_val)
        test_dataset = DGADataset(X_test, y_test)
        
        # 数据集信息
        dataset_info = {
            'vocab_size': info['vocab_size'],
            'max_length': info.get('max_length', X_train.shape[1]),
            'num_classes': info['num_classes'],
            'class_names': [f'class_{i}' for i in range(info['num_classes'])],
            'total_samples': info['total_samples'],
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'class_distribution': np.bincount(y_train)
        }
        
    elif 'X_train' in dataset:
        # 旧格式的多分类数据集
        X_train, y_train = dataset['X_train'], dataset['y_train']
        X_val, y_val = dataset['X_val'], dataset['y_val']
        X_test, y_test = dataset['X_test'], dataset['y_test']
        
        # 创建数据集对象
        train_dataset = DGADataset(X_train, y_train)
        val_dataset = DGADataset(X_val, y_val)
        test_dataset = DGADataset(X_test, y_test)
        
        # 数据集信息
        dataset_info = {
            'vocab_size': dataset['vocab_size'],
            'max_length': dataset.get('max_length', X_train.shape[1]),
            'num_classes': dataset.get('num_classes', 2),
            'class_names': dataset.get('class_names', ['benign', 'malicious']),
            'total_samples': len(X_train) + len(X_val) + len(X_test),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'class_distribution': np.bincount(y_train)
        }
        
    else:
        # 二分类数据集需要分割
        X, y = dataset['X'], dataset['y']
        
        # 创建数据集对象
        full_dataset = DGADataset(X, y)
        
        # 计算划分大小
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # 数据集划分
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        # 数据集信息
        dataset_info = {
            'vocab_size': dataset['vocab_size'],
            'max_length': dataset.get('max_length', X.shape[1]),
            'num_classes': 2,
            'class_names': ['benign', 'malicious'],
            'total_samples': total_size,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'class_distribution': np.bincount(y)
        }
    
    # 创建数据加载器 - 优化GPU利用率
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # 多进程数据加载
        pin_memory=True,  # 内存固定，加速GPU传输
        persistent_workers=True  # 保持工作进程活跃
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader, dataset_info


def print_dataset_info(dataset_info: Dict[str, Any]):
    """打印数据集信息"""
    print(f"📊 数据集信息:")
    print(f"  总样本数: {dataset_info['total_samples']}")
    print(f"  词汇表大小: {dataset_info['vocab_size']}")
    print(f"  最大序列长度: {dataset_info['max_length']}")
    print(f"  类别分布: {dataset_info['class_distribution']}")
    print(f"  训练集: {dataset_info['train_samples']}")
    print(f"  验证集: {dataset_info['val_samples']}")
    print(f"  测试集: {dataset_info['test_samples']}")