#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 数据集构建器
用于构建小规模的训练和测试数据集
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random
from collections import Counter
import pickle

class DGADatasetBuilder:
    """DGA数据集构建器"""
    
    def __init__(self, data_dir: str = "./data", max_length: int = 60):
        """
        初始化数据集构建器
        
        Args:
            data_dir: 数据目录路径
            max_length: 域名最大长度
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.char_to_idx = {}  # 字符到索引的映射
        self.idx_to_char = {}  # 索引到字符的映射
        self.vocab_size = 0
        
        # 初始化字符映射
        self._build_char_mapping()
    
    def _build_char_mapping(self):
        """构建字符到索引的映射"""
        # 定义字符集：小写字母、数字、特殊字符
        chars = list('abcdefghijklmnopqrstuvwxyz0123456789-._')
        chars.append('<PAD>')  # 添加填充字符
        
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.vocab_size = len(chars)
        
        print(f"字符集大小: {self.vocab_size}")
        print(f"字符集: {''.join(chars)}")
    
    def load_dga_data(self, family_name: str, num_samples: int = 1000) -> List[str]:
        """
        加载指定DGA家族的数据
        
        Args:
            family_name: DGA家族名称
            num_samples: 采样数量
            
        Returns:
            域名列表
        """
        file_path = os.path.join(self.data_dir, "DGA_Botnets_Domains", f"{family_name}.txt")
        
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            return []
        
        domains = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # 随机采样
                if len(lines) > num_samples:
                    lines = random.sample(lines, num_samples)
                
                for line in lines:
                    domain = line.strip().lower()
                    # 去除顶级域名后缀
                    if '.' in domain:
                        domain = domain.split('.')[0]
                    
                    # 过滤长度
                    if 3 <= len(domain) <= self.max_length:
                        domains.append(domain)
        
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        
        print(f"从 {family_name} 加载了 {len(domains)} 个域名")
        return domains
    
    def load_benign_data(self, num_samples: int = 1000) -> List[str]:
        """
        加载良性域名数据
        
        Args:
            num_samples: 采样数量
            
        Returns:
            域名列表
        """
        file_path = os.path.join(self.data_dir, "DGA_Botnets_Domains", "legit-1000000.txt")
        
        domains = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # 随机采样
                if len(lines) > num_samples:
                    lines = random.sample(lines, num_samples)
                
                for line in lines:
                    domain = line.strip().lower()
                    # 去除顶级域名后缀
                    if '.' in domain:
                        domain = domain.split('.')[0]
                    
                    # 过滤长度
                    if 3 <= len(domain) <= self.max_length:
                        domains.append(domain)
        
        except Exception as e:
            print(f"读取良性域名文件时出错: {e}")
        
        print(f"加载了 {len(domains)} 个良性域名")
        return domains
    
    def encode_domain(self, domain: str) -> List[int]:
        """
        将域名编码为数字序列
        
        Args:
            domain: 域名字符串
            
        Returns:
            编码后的数字列表
        """
        encoded = []
        for char in domain:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # 未知字符用下划线代替
                encoded.append(self.char_to_idx['_'])
        
        # 填充到固定长度
        pad_idx = self.char_to_idx['<PAD>']
        if len(encoded) < self.max_length:
            encoded.extend([pad_idx] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
        
        return encoded
    
    def build_small_dataset(self, 
                          dga_families: List[str] = None, 
                          samples_per_family: int = 500,
                          benign_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        构建小规模数据集
        
        Args:
            dga_families: DGA家族列表
            samples_per_family: 每个家族的样本数
            benign_samples: 良性样本数
            
        Returns:
            X: 特征矩阵
            y: 标签数组  
            families: 家族名称列表
        """
        if dga_families is None:
            # 选择几个主要的DGA家族
            dga_families = [
                'cryptolocker-50000',
                'necurs-50000', 
                'locky-50000',
                'ramnit-50000',
                'zeus-50000'
            ]
        
        all_domains = []
        all_labels = []
        family_labels = []
        
        # 加载良性域名 (标签为0)
        print("加载良性域名...")
        benign_domains = self.load_benign_data(benign_samples)
        all_domains.extend(benign_domains)
        all_labels.extend([0] * len(benign_domains))
        family_labels.extend(['benign'] * len(benign_domains))
        
        # 加载DGA域名
        for i, family in enumerate(dga_families):
            print(f"加载DGA家族: {family}")
            dga_domains = self.load_dga_data(family, samples_per_family)
            all_domains.extend(dga_domains)
            all_labels.extend([1] * len(dga_domains))  # DGA标签为1
            family_labels.extend([family] * len(dga_domains))
        
        print(f"\n数据集统计:")
        print(f"总样本数: {len(all_domains)}")
        print(f"良性样本: {sum(1 for x in all_labels if x == 0)}")
        print(f"恶意样本: {sum(1 for x in all_labels if x == 1)}")
        
        # 编码域名
        print("编码域名...")
        X = []
        valid_indices = []
        
        for i, domain in enumerate(all_domains):
            if domain:  # 确保域名非空
                encoded = self.encode_domain(domain)
                X.append(encoded)
                valid_indices.append(i)
        
        # 过滤有效数据
        X = np.array(X)
        y = np.array([all_labels[i] for i in valid_indices])
        families = [family_labels[i] for i in valid_indices]
        
        print(f"最终数据集形状: X={X.shape}, y={y.shape}")
        return X, y, families
    
    def analyze_dataset(self, domains: List[str], labels: List[int]):
        """分析数据集统计信息"""
        print("\n=== 数据集分析 ===")
        
        # 标签分布
        label_counts = Counter(labels)
        print(f"标签分布: {dict(label_counts)}")
        
        # 长度统计
        lengths = [len(domain) for domain in domains]
        print(f"域名长度统计:")
        print(f"  最小长度: {min(lengths)}")
        print(f"  最大长度: {max(lengths)}")
        print(f"  平均长度: {np.mean(lengths):.2f}")
        print(f"  标准差: {np.std(lengths):.2f}")
        
        # 字符统计
        all_chars = ''.join(domains)
        char_counts = Counter(all_chars)
        print(f"字符种类数: {len(char_counts)}")
        print(f"最常见字符: {char_counts.most_common(10)}")
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, families: List[str], 
                    file_path: str = "./data/small_dataset.pkl"):
        """保存数据集"""
        dataset = {
            'X': X,
            'y': y, 
            'families': families,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"数据集已保存到: {file_path}")
    
    def load_dataset(self, file_path: str = "./data/small_dataset.pkl") -> Dict:
        """加载数据集"""
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # 恢复字符映射
        self.char_to_idx = dataset['char_to_idx']
        self.idx_to_char = dataset['idx_to_char'] 
        self.vocab_size = dataset['vocab_size']
        self.max_length = dataset['max_length']
        
        print(f"数据集已从 {file_path} 加载")
        print(f"数据形状: X={dataset['X'].shape}, y={dataset['y'].shape}")
        
        return dataset


def main():
    """主函数 - 构建小规模数据集"""
    print("=== DGA恶意域名检测 - 小规模数据集构建 ===\n")
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建数据集构建器
    builder = DGADatasetBuilder(data_dir="./data", max_length=40)
    
    # 构建小规模数据集
    print("构建小规模数据集...")
    X, y, families = builder.build_small_dataset(
        dga_families=[
            'cryptolocker-50000',
            'necurs-50000', 
            'locky-50000',
            'ramnit-50000'
        ],
        samples_per_family=300,  # 每个DGA家族300个样本
        benign_samples=1200      # 1200个良性样本，保持平衡
    )
    
    # 分析数据集
    domains_for_analysis = []
    for i in range(min(1000, len(X))):  # 分析前1000个样本
        domain = ''.join([builder.idx_to_char.get(idx, '') for idx in X[i] if idx != builder.char_to_idx['<PAD>']])
        domains_for_analysis.append(domain)
    
    builder.analyze_dataset(domains_for_analysis, y[:len(domains_for_analysis)].tolist())
    
    # 保存数据集
    builder.save_dataset(X, y, families, "./data/small_dga_dataset.pkl")
    
    print("\n数据集构建完成！")


if __name__ == "__main__":
    main()