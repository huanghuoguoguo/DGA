#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA恶意域名检测 - 大规模数据集构建器
用于构建更大规模的训练和测试数据集，验证数据量对MoE模型性能的影响
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random
from collections import Counter
import pickle
import math

class LargeDGADatasetBuilder:
    """大规模DGA数据集构建器"""
    
    def __init__(self, data_dir: str = "./data", max_length: int = 40):
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
        
        # 可用的DGA家族列表
        self.available_families = [
            'cryptolocker-50000', 'necurs-50000', 'locky-50000', 'ramnit-50000',
            'bamital-50000', 'banjori-50000', 'bazarbackdoor-50000', 'bedep-50000',
            'bigviktor-50000', 'chinad-50000', 'corebot-50000', 'dircrypt-50000',
            'dnschanger-50000', 'dyre-50000', 'emotet-50000', 'enviserv-50000',
            'fobber_v1-50000', 'gozi_gpl-50000', 'kraken_v1-50000', 'matsnu-50000',
            'monerodownloader-50000', 'murofet_v1-50000', 'murofetweekly-50000',
            'mydoom-50000', 'newgoz-50000', 'nymaim2-50000', 'padcrypt-50000',
            'pandabanker-20000', 'pitou-50000', 'proslikefan-50000', 'pushdo-50000',
            'pykspa_improved_useful-50000', 'pykspa_precursor-50000', 'qadars-50000',
            'qakbot-50000', 'qsnatch-50000', 'ramdo-50000', 'ranbyus_v1-50000',
            'reconyc-50000', 'rovnix-50000', 'shiotob-50000', 'simda-50000',
            'sisron-50000', 'sphinx-50000', 'suppobox_1-20000', 'symmi-50000',
            'tempedreve-50000', 'tinba-50000', 'tinynuke-50000', 'torpig-50000',
            'vawtrak_v1-50000', 'vidro-50000', 'virut-50000', 'wd-20000', 'zloader-50000'
        ]
        
    def _build_char_mapping(self):
        """构建字符映射表"""
        # 基础字符集：a-z, 0-9, 特殊字符
        chars = ['<PAD>', '<UNK>'] + list('abcdefghijklmnopqrstuvwxyz0123456789-_.')
        
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.vocab_size = len(chars)
        
        print(f"字符映射表构建完成，词汇表大小: {self.vocab_size}")
    
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
                    
                    # 过滤长度和字符
                    if 3 <= len(domain) <= self.max_length and self._is_valid_domain(domain):
                        domains.append(domain)
        
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        
        print(f"从 {family_name} 加载了 {len(domains)} 个域名")
        return domains
    
    def _is_valid_domain(self, domain: str) -> bool:
        """检查域名是否有效"""
        # 只包含允许的字符
        allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789-_.')
        return all(c in allowed_chars for c in domain)
    
    def load_benign_data(self, num_samples: int = 1000) -> List[str]:
        """
        加载良性域名数据
        
        Args:
            num_samples: 采样数量
            
        Returns:
            良性域名列表
        """
        benign_files = [
            os.path.join(self.data_dir, "DGA_Botnets_Domains", "legit-1000000.txt"),
            os.path.join(self.data_dir, "alexa_1000000"),
            os.path.join(self.data_dir, "alexa_2019"),
            os.path.join(self.data_dir, "top-1m.txt")
        ]
        
        domains = []
        samples_per_file = num_samples // len([f for f in benign_files if os.path.exists(f)])
        
        for file_path in benign_files:
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if len(lines) > samples_per_file:
                        lines = random.sample(lines, samples_per_file)
                    
                    for line in lines:
                        domain = line.strip().lower()
                        # 处理不同格式的文件
                        if ',' in domain:  # Alexa格式: rank,domain
                            domain = domain.split(',')[1] if len(domain.split(',')) > 1 else domain.split(',')[0]
                        
                        # 去除顶级域名后缀
                        if '.' in domain:
                            domain = domain.split('.')[0]
                        
                        # 过滤长度和字符
                        if 3 <= len(domain) <= self.max_length and self._is_valid_domain(domain):
                            domains.append(domain)
                            
                        if len(domains) >= num_samples:
                            break
                            
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue
        
        # 如果样本不足，随机重复一些
        while len(domains) < num_samples and domains:
            domains.extend(random.sample(domains, min(len(domains), num_samples - len(domains))))
        
        domains = domains[:num_samples]  # 确保不超过要求数量
        print(f"加载了 {len(domains)} 个良性域名")
        return domains
    
    def encode_domain(self, domain: str) -> List[int]:
        """
        将域名编码为整数序列
        
        Args:
            domain: 域名字符串
            
        Returns:
            编码后的整数列表
        """
        encoded = []
        for char in domain:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.char_to_idx['<UNK>'])  # 未知字符
        
        # 填充或截断到固定长度
        if len(encoded) < self.max_length:
            encoded.extend([self.char_to_idx['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
        
        return encoded
    
    def build_large_dataset(self, 
                           target_size: int = 20000,
                           dga_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        构建大规模数据集
        
        Args:
            target_size: 目标数据集大小
            dga_ratio: DGA样本比例
            
        Returns:
            X: 特征矩阵
            y: 标签数组  
            families: 家族名称列表
        """
        dga_samples = int(target_size * dga_ratio)
        benign_samples = target_size - dga_samples
        
        print(f"构建大规模数据集: 总样本{target_size}, DGA样本{dga_samples}, 良性样本{benign_samples}")
        
        all_domains = []
        all_labels = []
        family_labels = []
        
        # 加载良性域名
        print("\n加载良性域名...")
        benign_domains = self.load_benign_data(benign_samples)
        all_domains.extend(benign_domains)
        all_labels.extend([0] * len(benign_domains))
        family_labels.extend(['benign'] * len(benign_domains))
        
        # 计算每个DGA家族需要的样本数
        available_families = [f for f in self.available_families 
                            if os.path.exists(os.path.join(self.data_dir, "DGA_Botnets_Domains", f"{f}.txt"))]
        
        samples_per_family = max(1, dga_samples // len(available_families))
        remaining_samples = dga_samples - (samples_per_family * len(available_families))
        
        print(f"\n可用DGA家族: {len(available_families)}")
        print(f"每个家族样本数: {samples_per_family}")
        
        # 加载DGA域名
        for i, family in enumerate(available_families):
            # 为前几个家族分配额外的样本
            current_samples = samples_per_family
            if i < remaining_samples:
                current_samples += 1
                
            print(f"加载DGA家族: {family} ({current_samples}个样本)")
            dga_domains = self.load_dga_data(family, current_samples)
            
            all_domains.extend(dga_domains)
            all_labels.extend([1] * len(dga_domains))  # DGA标签为1
            family_labels.extend([family] * len(dga_domains))
        
        print(f"\n数据集统计:")
        print(f"总样本数: {len(all_domains)}")
        print(f"良性样本: {sum(1 for x in all_labels if x == 0)}")
        print(f"恶意样本: {sum(1 for x in all_labels if x == 1)}")
        
        # 编码域名
        print("\n编码域名...")
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
    
    def analyze_large_dataset(self, domains: List[str], labels: List[int], families: List[str]):
        """分析大规模数据集的特征"""
        print("\n=== 大规模数据集分析 ===")
        
        # 基本统计
        total_samples = len(domains)
        benign_count = sum(1 for label in labels if label == 0)
        dga_count = sum(1 for label in labels if label == 1)
        
        print(f"总样本数: {total_samples}")
        print(f"良性样本: {benign_count} ({benign_count/total_samples*100:.1f}%)")
        print(f"恶意样本: {dga_count} ({dga_count/total_samples*100:.1f}%)")
        
        # 长度分析
        lengths = [len(domain) for domain in domains]
        print(f"\n域名长度统计:")
        print(f"平均长度: {np.mean(lengths):.2f}")
        print(f"最短长度: {min(lengths)}")
        print(f"最长长度: {max(lengths)}")
        print(f"长度标准差: {np.std(lengths):.2f}")
        
        # DGA家族分析
        dga_families = [f for f, label in zip(families, labels) if label == 1 and f != 'benign']
        family_counts = Counter(dga_families)
        
        print(f"\nDGA家族分布 (前10):")
        for family, count in family_counts.most_common(10):
            print(f"  {family}: {count}个样本")
        
        print(f"总DGA家族数: {len(family_counts)}")
        
        # 字符分析
        all_chars = set(''.join(domains))
        print(f"\n字符集分析:")
        print(f"唯一字符数: {len(all_chars)}")
        print(f"字符集: {''.join(sorted(all_chars))}")
        
        # 熵分析（简化版）
        def calculate_entropy(domain):
            char_counts = Counter(domain)
            total_chars = len(domain)
            entropy = 0
            for count in char_counts.values():
                p = count / total_chars
                entropy -= p * math.log2(p)
            return entropy
        
        benign_entropies = [calculate_entropy(domain) for domain, label in zip(domains, labels) if label == 0]
        dga_entropies = [calculate_entropy(domain) for domain, label in zip(domains, labels) if label == 1]
        
        print(f"\n熵分析:")
        print(f"良性域名平均熵: {np.mean(benign_entropies):.3f}")
        print(f"DGA域名平均熵: {np.mean(dga_entropies):.3f}")
        print(f"熵差异: {np.mean(dga_entropies) - np.mean(benign_entropies):.3f}")
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, families: List[str], 
                    file_path: str = "./data/large_dga_dataset.pkl"):
        """保存数据集到文件"""
        dataset = {
            'X': X,
            'y': y,
            'families': families,
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'max_length': self.max_length
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n数据集已保存到: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    
    def load_dataset(self, file_path: str = "./data/large_dga_dataset.pkl") -> Dict:
        """从文件加载数据集"""
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset


def main():
    """主函数 - 构建大规模数据集"""
    print("=== DGA恶意域名检测 - 大规模数据集构建 ===\n")
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建数据集构建器
    builder = LargeDGADatasetBuilder(data_dir="./data", max_length=40)
    
    # 构建不同规模的数据集
    dataset_configs = [
        {"name": "中等规模", "size": 10000, "filename": "medium_dga_dataset.pkl"},
        {"name": "大规模", "size": 20000, "filename": "large_dga_dataset.pkl"},
        {"name": "超大规模", "size": 50000, "filename": "xlarge_dga_dataset.pkl"}
    ]
    
    for config in dataset_configs:
        print(f"\n{'='*60}")
        print(f"构建{config['name']}数据集 ({config['size']}样本)")
        print(f"{'='*60}")
        
        try:
            X, y, families = builder.build_large_dataset(
                target_size=config['size'],
                dga_ratio=0.5  # 保持平衡
            )
            
            # 分析数据集
            domains_for_analysis = []
            for i in range(min(1000, len(X))):
                domain = ''.join([builder.idx_to_char.get(idx, '') for idx in X[i] 
                                if idx != builder.char_to_idx['<PAD>']])
                domains_for_analysis.append(domain)
            
            builder.analyze_large_dataset(
                domains_for_analysis, 
                y[:len(domains_for_analysis)].tolist(),
                families[:len(domains_for_analysis)]
            )
            
            # 保存数据集
            save_path = f"./data/processed/{config['filename']}"
            builder.save_dataset(X, y, families, save_path)
            
            print(f"\n✅ {config['name']}数据集构建完成！")
            
        except Exception as e:
            print(f"❌ 构建{config['name']}数据集时出错: {e}")
            continue
    
    print(f"\n🎉 所有数据集构建完成！")
    print(f"\n📝 使用方法:")
    print(f"  python simple_train.py --model simplified_moe --dataset medium")
    print(f"  python simple_train.py --model simplified_moe --dataset large")
    print(f"  python simple_train.py --model simplified_moe --dataset xlarge")


if __name__ == "__main__":
    main()