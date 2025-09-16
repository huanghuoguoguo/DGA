#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 多分类数据集构建器
用于构建多分类的训练和测试数据集，区分不同的DGA家族
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split

class MultiClassDGADatasetBuilder:
    """多分类DGA数据集构建器"""
    
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
        self.class_to_idx = {}  # 类别到索引的映射
        self.idx_to_class = {}  # 索引到类别的映射
        self.num_classes = 0
        
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
    
    def _build_class_mapping(self, class_names: List[str]):
        """构建类别到索引的映射"""
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(class_names)}
        self.num_classes = len(class_names)
        
        print(f"\n类别映射:")
        for cls, idx in self.class_to_idx.items():
            print(f"  {idx}: {cls}")
    
    def generate_synthetic_dga_data(self, family_name: str, num_samples: int = 1000) -> List[str]:
        """生成合成DGA数据"""
        domains = []
        
        # 不同DGA家族的生成策略
        if 'conficker' in family_name.lower():
            # Conficker: 短随机字符串
            for _ in range(num_samples):
                length = random.randint(6, 12)
                domain = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
                domains.append(domain)
                
        elif 'zeus' in family_name.lower():
            # Zeus: 混合字母数字
            for _ in range(num_samples):
                length = random.randint(8, 16)
                chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
                domain = ''.join(random.choices(chars, k=length))
                domains.append(domain)
                
        elif 'necurs' in family_name.lower():
            # Necurs: 长随机字符串
            for _ in range(num_samples):
                length = random.randint(12, 20)
                domain = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
                domains.append(domain)
                
        elif 'locky' in family_name.lower():
            # Locky: 短字符串，偏向特定字符
            chars = 'abcdefghijklmnopqrstuvwxyz'
            weights = [1 if c in 'aeiou' else 2 for c in chars]  # 偏向辅音
            for _ in range(num_samples):
                length = random.randint(6, 10)
                domain = ''.join(random.choices(chars, weights=weights, k=length))
                domains.append(domain)
                
        elif 'ramnit' in family_name.lower():
            # Ramnit: 元音辅音交替模式
            vowels = 'aeiou'
            consonants = 'bcdfghjklmnpqrstvwxyz'
            for _ in range(num_samples):
                length = random.randint(8, 14)
                domain = ''
                for i in range(length):
                    if i % 2 == 0:
                        domain += random.choice(consonants)
                    else:
                        domain += random.choice(vowels)
                domains.append(domain)
                
        elif 'cryptolocker' in family_name.lower():
            # Cryptolocker: 复杂混合模式
            for _ in range(num_samples):
                length = random.randint(10, 18)
                # 前半部分字母，后半部分数字
                letters_part = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length//2))
                numbers_part = ''.join(random.choices('0123456789', k=length-length//2))
                domain = letters_part + numbers_part
                domains.append(domain)
                
        else:
            # 默认：随机字符串
            for _ in range(num_samples):
                length = random.randint(8, 16)
                domain = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
                domains.append(domain)
        
        return domains
    
    def generate_benign_data(self, num_samples: int = 1000) -> List[str]:
        """生成合成良性域名数据"""
        # 常见的良性域名模式
        common_words = [
            'google', 'facebook', 'amazon', 'microsoft', 'apple', 'twitter',
            'youtube', 'instagram', 'linkedin', 'github', 'stackoverflow',
            'wikipedia', 'reddit', 'netflix', 'spotify', 'dropbox', 'zoom',
            'slack', 'discord', 'telegram', 'whatsapp', 'skype', 'adobe',
            'oracle', 'salesforce', 'shopify', 'paypal', 'stripe', 'square'
        ]
        
        suffixes = ['app', 'web', 'site', 'online', 'cloud', 'tech', 'pro', 'net']
        prefixes = ['my', 'get', 'use', 'try', 'go', 'new', 'best', 'top']
        
        domains = []
        
        for _ in range(num_samples):
            if random.random() < 0.4:
                # 直接使用常见词汇
                domain = random.choice(common_words)
            elif random.random() < 0.3:
                # 前缀 + 常见词汇
                domain = random.choice(prefixes) + random.choice(common_words)
            elif random.random() < 0.2:
                # 常见词汇 + 后缀
                domain = random.choice(common_words) + random.choice(suffixes)
            else:
                # 前缀 + 常见词汇 + 后缀
                domain = random.choice(prefixes) + random.choice(common_words) + random.choice(suffixes)
            
            domains.append(domain.lower())
        
        return domains
    
    def encode_domain(self, domain: str) -> List[int]:
        """将域名编码为整数序列"""
        # 转换为小写并限制长度
        domain = domain.lower()[:self.max_length]
        
        # 编码每个字符
        encoded = []
        for char in domain:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # 未知字符用下划线代替
                encoded.append(self.char_to_idx.get('_', 0))
        
        # 填充到固定长度
        pad_idx = self.char_to_idx['<PAD>']
        while len(encoded) < self.max_length:
            encoded.append(pad_idx)
        
        return encoded
    
    def build_multiclass_dataset(self, 
                               dga_families: List[str] = None,
                               samples_per_family: int = 500,
                               benign_samples: int = 1000,
                               include_benign: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        构建多分类数据集
        
        Args:
            dga_families: DGA家族列表
            samples_per_family: 每个家族的样本数
            benign_samples: 良性样本数
            include_benign: 是否包含良性类别
            
        Returns:
            X: 特征矩阵
            y: 标签数组  
            class_names: 类别名称列表
        """
        if dga_families is None:
            # 选择几个主要的DGA家族
            dga_families = [
                'conficker',
                'zeus', 
                'necurs',
                'locky',
                'ramnit',
                'cryptolocker'
            ]
        
        # 构建类别映射
        class_names = []
        if include_benign:
            class_names.append('benign')
        class_names.extend(dga_families)
        
        self._build_class_mapping(class_names)
        
        all_domains = []
        all_labels = []
        
        # 加载良性域名
        if include_benign:
            print("生成良性域名...")
            benign_domains = self.generate_benign_data(benign_samples)
            all_domains.extend(benign_domains)
            all_labels.extend([self.class_to_idx['benign']] * len(benign_domains))
        
        # 加载DGA域名
        for family in dga_families:
            print(f"生成DGA家族: {family}")
            dga_domains = self.generate_synthetic_dga_data(family, samples_per_family)
            all_domains.extend(dga_domains)
            all_labels.extend([self.class_to_idx[family]] * len(dga_domains))
        
        print(f"\n数据集统计:")
        print(f"总样本数: {len(all_domains)}")
        for class_name in class_names:
            count = sum(1 for x in all_labels if x == self.class_to_idx[class_name])
            print(f"{class_name}: {count}")
        
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
        
        print(f"最终数据集形状: X={X.shape}, y={y.shape}")
        print(f"类别数: {self.num_classes}")
        
        return X, y, class_names
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2, val_size: float = 0.1, 
                     random_state: int = 42) -> Tuple:
        """分割数据集为训练、验证、测试集"""
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 从剩余数据中分离出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\n数据集分割:")
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def analyze_dataset(self, domains: List[str], labels: List[int]):
        """分析数据集统计信息"""
        print("\n=== 数据集分析 ===")
        
        # 标签分布
        label_counts = Counter(labels)
        print(f"标签分布:")
        for label, count in sorted(label_counts.items()):
            class_name = self.idx_to_class.get(label, f"Class_{label}")
            print(f"  {class_name}: {count}")
        
        # 长度统计
        lengths = [len(domain) for domain in domains]
        print(f"\n域名长度统计:")
        print(f"  最小长度: {min(lengths)}")
        print(f"  最大长度: {max(lengths)}")
        print(f"  平均长度: {np.mean(lengths):.2f}")
        print(f"  标准差: {np.std(lengths):.2f}")
        
        # 字符统计
        all_chars = ''.join(domains)
        char_counts = Counter(all_chars)
        print(f"\n字符种类数: {len(char_counts)}")
        print(f"最常见字符: {char_counts.most_common(10)}")
    
    def save_dataset(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                    class_names: List[str], file_path: str = "./data/processed/multiclass_dga_dataset.pkl"):
        """保存多分类数据集"""
        dataset = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'class_names': class_names,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': self.num_classes,
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'max_length': self.max_length
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n数据集已保存到: {file_path}")
        print(f"数据集信息:")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")
        print(f"  类别数: {self.num_classes}")
        print(f"  词汇表大小: {self.vocab_size}")
    
    def load_dataset(self, file_path: str = "./data/processed/multiclass_dga_dataset.pkl") -> Dict:
        """加载数据集"""
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # 恢复映射
        self.class_to_idx = dataset['class_to_idx']
        self.idx_to_class = dataset['idx_to_class']
        self.num_classes = dataset['num_classes']
        self.char_to_idx = dataset['char_to_idx']
        self.idx_to_char = dataset['idx_to_char']
        self.vocab_size = dataset['vocab_size']
        self.max_length = dataset['max_length']
        
        return dataset

def main():
    """主函数"""
    print("=== DGA多分类数据集构建器 ===")
    
    # 创建构建器
    builder = MultiClassDGADatasetBuilder()
    
    # 构建小规模多分类数据集
    print("\n构建小规模多分类数据集...")
    X, y, class_names = builder.build_multiclass_dataset(
        dga_families=['conficker', 'zeus', 'necurs', 'locky'],
        samples_per_family=300,
        benign_samples=400,
        include_benign=True
    )
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = builder.split_dataset(X, y)
    
    # 保存数据集
    builder.save_dataset(
        X_train, X_val, X_test, y_train, y_val, y_test, class_names,
        "./data/processed/small_multiclass_dga_dataset.pkl"
    )
    
    # 构建中等规模多分类数据集
    print("\n构建中等规模多分类数据集...")
    X, y, class_names = builder.build_multiclass_dataset(
        dga_families=['conficker', 'zeus', 'necurs', 'locky', 'ramnit', 'cryptolocker'],
        samples_per_family=800,
        benign_samples=1000,
        include_benign=True
    )
    
    # 分割数据集
    X_train, X_val, X_test, y_train, y_val, y_test = builder.split_dataset(X, y)
    
    # 保存数据集
    builder.save_dataset(
        X_train, X_val, X_test, y_train, y_val, y_test, class_names,
        "./data/processed/medium_multiclass_dga_dataset.pkl"
    )
    
    print("\n多分类数据集构建完成！")

if __name__ == "__main__":
    main()