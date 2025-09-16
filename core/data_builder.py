#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据集构建器
生成格式统一的二分类和多分类DGA数据集
支持小型(1万)和大型(10万)数据集
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import string
from typing import Dict, List, Tuple, Any


class UnifiedDGADatasetBuilder:
    """统一DGA数据集构建器"""
    
    def __init__(self):
        # 常见TLD
        self.tlds = ['.com', '.net', '.org', '.info', '.biz', '.co', '.io', '.me', '.tv', '.cc']
        
        # DGA家族配置 - 15个主要DGA家族
        self.dga_families = {
            'conficker': {'weight': 0.12, 'min_len': 8, 'max_len': 15, 'pattern': 'random'},
            'cryptolocker': {'weight': 0.10, 'min_len': 12, 'max_len': 20, 'pattern': 'mixed'},
            'zeus': {'weight': 0.09, 'min_len': 10, 'max_len': 16, 'pattern': 'dictionary'},
            'necurs': {'weight': 0.08, 'min_len': 8, 'max_len': 12, 'pattern': 'random'},
            'matsnu': {'weight': 0.07, 'min_len': 6, 'max_len': 10, 'pattern': 'dictionary'},
            'suppobox': {'weight': 0.07, 'min_len': 8, 'max_len': 14, 'pattern': 'mixed'},
            'tinba': {'weight': 0.06, 'min_len': 7, 'max_len': 12, 'pattern': 'random'},
            'ramdo': {'weight': 0.06, 'min_len': 9, 'max_len': 15, 'pattern': 'mixed'},
            'banjori': {'weight': 0.06, 'min_len': 8, 'max_len': 12, 'pattern': 'random'},
            'corebot': {'weight': 0.05, 'min_len': 10, 'max_len': 18, 'pattern': 'dictionary'},
            'dircrypt': {'weight': 0.05, 'min_len': 12, 'max_len': 20, 'pattern': 'mixed'},
            'kraken': {'weight': 0.05, 'min_len': 8, 'max_len': 14, 'pattern': 'random'},
            'locky': {'weight': 0.05, 'min_len': 16, 'max_len': 25, 'pattern': 'mixed'},
            'pykspa': {'weight': 0.04, 'min_len': 6, 'max_len': 10, 'pattern': 'dictionary'},
            'qakbot': {'weight': 0.05, 'min_len': 8, 'max_len': 16, 'pattern': 'random'}
        }
        
        # 字典单词（用于生成字典型DGA）
        self.dictionary_words = [
            'secure', 'bank', 'login', 'account', 'service', 'update', 'verify', 'confirm',
            'support', 'help', 'mail', 'web', 'site', 'page', 'home', 'user', 'admin',
            'system', 'network', 'server', 'client', 'data', 'info', 'news', 'blog',
            'shop', 'store', 'buy', 'sell', 'pay', 'money', 'cash', 'card', 'credit',
            'online', 'digital', 'tech', 'soft', 'app', 'mobile', 'cloud', 'smart',
            'fast', 'quick', 'easy', 'simple', 'best', 'top', 'new', 'free', 'safe'
        ]
        
        # 良性域名模式
        self.benign_patterns = [
            'company', 'business', 'service', 'tech', 'digital', 'solutions', 'systems',
            'global', 'international', 'world', 'group', 'corp', 'inc', 'ltd', 'llc'
        ]
    
    def generate_random_string(self, min_len: int, max_len: int, charset: str = None) -> str:
        """生成随机字符串"""
        if charset is None:
            charset = string.ascii_lowercase
        length = random.randint(min_len, max_len)
        return ''.join(random.choice(charset) for _ in range(length))
    
    def generate_dictionary_domain(self, min_len: int, max_len: int) -> str:
        """生成基于字典的域名"""
        words = random.sample(self.dictionary_words, random.randint(1, 3))
        domain = ''.join(words)
        
        # 如果太长，截断
        if len(domain) > max_len:
            domain = domain[:max_len]
        
        # 如果太短，添加随机字符
        while len(domain) < min_len:
            domain += random.choice(string.ascii_lowercase)
        
        return domain
    
    def generate_mixed_domain(self, min_len: int, max_len: int) -> str:
        """生成混合型域名（字典+随机）"""
        # 随机选择1-2个字典单词
        words = random.sample(self.dictionary_words, random.randint(1, 2))
        base = ''.join(words)
        
        # 添加随机字符或数字
        remaining_len = random.randint(max(0, min_len - len(base)), max_len - len(base))
        if remaining_len > 0:
            charset = string.ascii_lowercase + string.digits
            random_part = ''.join(random.choice(charset) for _ in range(remaining_len))
            
            # 随机决定随机部分的位置
            if random.choice([True, False]):
                domain = base + random_part
            else:
                domain = random_part + base
        else:
            domain = base
        
        # 确保长度在范围内
        if len(domain) > max_len:
            domain = domain[:max_len]
        while len(domain) < min_len:
            domain += random.choice(string.ascii_lowercase)
        
        return domain
    
    def generate_dga_domain(self, family_name: str, family_config: Dict) -> str:
        """根据DGA家族生成恶意域名"""
        pattern = family_config['pattern']
        min_len = family_config['min_len']
        max_len = family_config['max_len']
        
        if pattern == 'random':
            domain = self.generate_random_string(min_len, max_len)
        elif pattern == 'dictionary':
            domain = self.generate_dictionary_domain(min_len, max_len)
        elif pattern == 'mixed':
            domain = self.generate_mixed_domain(min_len, max_len)
        else:
            domain = self.generate_random_string(min_len, max_len)
        
        # 添加TLD
        tld = random.choice(self.tlds)
        full_domain = domain + tld
        
        return full_domain
    
    def generate_benign_domain(self) -> str:
        """生成良性域名"""
        # 选择生成策略
        strategy = random.choice(['company', 'service', 'personal', 'tech'])
        
        if strategy == 'company':
            # 公司域名：company_name + suffix
            base = random.choice(self.benign_patterns)
            suffix = random.choice(['tech', 'corp', 'inc', 'group', 'solutions'])
            domain = base + suffix
        elif strategy == 'service':
            # 服务域名：service_type + descriptor
            service = random.choice(['mail', 'web', 'cloud', 'data', 'secure'])
            descriptor = random.choice(['service', 'system', 'platform', 'hub'])
            domain = service + descriptor
        elif strategy == 'personal':
            # 个人域名：name + number/suffix
            name = random.choice(['john', 'mike', 'sarah', 'alex', 'chris', 'david'])
            suffix = random.choice(['blog', 'site', 'home', str(random.randint(1, 999))])
            domain = name + suffix
        else:  # tech
            # 技术域名：tech_term + suffix
            tech = random.choice(['app', 'dev', 'code', 'soft', 'tech', 'digital'])
            suffix = random.choice(['lab', 'hub', 'zone', 'space', 'world'])
            domain = tech + suffix
        
        # 添加TLD
        tld = random.choice(self.tlds)
        full_domain = domain + tld
        
        return full_domain
    
    def build_binary_dataset(self, total_samples: int, benign_ratio: float = 0.5) -> pd.DataFrame:
        """构建二分类数据集"""
        print(f"构建二分类数据集 (总样本: {total_samples:,})...")
        
        benign_samples = int(total_samples * benign_ratio)
        malicious_samples = total_samples - benign_samples
        
        domains = []
        labels = []
        families = []
        
        # 生成良性域名
        print(f"生成 {benign_samples:,} 个良性域名...")
        for i in range(benign_samples):
            if (i + 1) % 2000 == 0:
                print(f"  已生成 {i+1:,} 个良性域名")
            
            domain = self.generate_benign_domain()
            domains.append(domain)
            labels.append(0)  # 良性标签为0
            families.append('benign')
        
        # 生成恶意域名（随机选择DGA家族）
        print(f"生成 {malicious_samples:,} 个恶意域名...")
        family_list = list(self.dga_families.keys())
        
        for i in range(malicious_samples):
            if (i + 1) % 2000 == 0:
                print(f"  已生成 {i+1:,} 个恶意域名")
            
            # 随机选择DGA家族
            family = random.choice(family_list)
            family_config = self.dga_families[family]
            
            domain = self.generate_dga_domain(family, family_config)
            domains.append(domain)
            labels.append(1)  # 恶意标签为1
            families.append(family)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'domain': domains,
            'label': labels,
            'family': families
        })
        
        # 打乱数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"二分类数据集构建完成: {len(df):,} 样本")
        return df
    
    def build_multiclass_dataset(self, total_samples: int, benign_ratio: float = 0.5) -> pd.DataFrame:
        """构建多分类数据集"""
        print(f"构建多分类数据集 (总样本: {total_samples:,})...")
        
        benign_samples = int(total_samples * benign_ratio)
        malicious_samples = total_samples - benign_samples
        
        domains = []
        labels = []
        families = []
        
        # 生成良性域名
        print(f"生成 {benign_samples:,} 个良性域名...")
        for i in range(benign_samples):
            if (i + 1) % 2000 == 0:
                print(f"  已生成 {i+1:,} 个良性域名")
            
            domain = self.generate_benign_domain()
            domains.append(domain)
            labels.append(0)  # 良性标签为0
            families.append('benign')
        
        # 生成恶意域名（按权重分配）
        print(f"生成 {malicious_samples:,} 个恶意域名...")
        family_list = list(self.dga_families.keys())
        family_weights = [self.dga_families[f]['weight'] for f in family_list]
        
        # 根据权重分配每个家族的样本数
        family_counts = {}
        remaining_samples = malicious_samples
        
        for i, family in enumerate(family_list[:-1]):
            count = int(malicious_samples * family_weights[i])
            family_counts[family] = count
            remaining_samples -= count
        
        # 最后一个家族获得剩余样本
        family_counts[family_list[-1]] = remaining_samples
        
        print(f"DGA家族样本分布:")
        for family, count in family_counts.items():
            print(f"  {family}: {count:,} 样本")
        
        # 生成每个家族的域名
        label_counter = 1  # 从1开始，0是良性
        for family, count in family_counts.items():
            family_config = self.dga_families[family]
            
            for i in range(count):
                if (i + 1) % 1000 == 0:
                    print(f"  已生成 {i+1:,} 个 {family} 域名")
                
                domain = self.generate_dga_domain(family, family_config)
                domains.append(domain)
                labels.append(label_counter)
                families.append(family)
            
            label_counter += 1
        
        # 创建DataFrame
        df = pd.DataFrame({
            'domain': domains,
            'label': labels,
            'family': families
        })
        
        # 打乱数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"多分类数据集构建完成: {len(df):,} 样本，{df['label'].nunique()} 类别")
        return df
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """分割数据集 (8:1:1)"""
        print(f"分割数据集 (训练:{train_ratio*100:.0f}%, 验证:{val_ratio*100:.0f}%, 测试:{test_ratio*100:.0f}%)...")
        
        # 确保比例总和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # 分层抽样，确保每个类别在各个集合中的比例相同
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio), 
            stratify=df['label'], random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio/(val_ratio + test_ratio),
            stratify=temp_df['label'], random_state=42
        )
        
        print(f"训练集: {len(train_df):,} 样本")
        print(f"验证集: {len(val_df):,} 样本")
        print(f"测试集: {len(test_df):,} 样本")
        
        return train_df, val_df, test_df
    
    def create_unified_format(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                            test_df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
        """创建统一格式的数据集"""
        print("创建统一格式数据集...")
        
        # 获取所有字符
        all_chars = set()
        for domain in train_df['domain']:
            all_chars.update(domain.lower())
        
        # 添加特殊字符
        all_chars.add('<PAD>')  # 填充
        all_chars.add('<UNK>')  # 未知字符
        
        char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # 转换域名为数字序列
        def domain_to_sequence(domain: str, max_len: int = 40) -> List[int]:
            sequence = [char_to_idx.get(char.lower(), char_to_idx['<UNK>']) for char in domain]
            # 截断或填充
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            else:
                sequence.extend([char_to_idx['<PAD>']] * (max_len - len(sequence)))
            return sequence
        
        # 处理数据
        max_length = 40
        
        train_sequences = [domain_to_sequence(domain, max_length) for domain in train_df['domain']]
        val_sequences = [domain_to_sequence(domain, max_length) for domain in val_df['domain']]
        test_sequences = [domain_to_sequence(domain, max_length) for domain in test_df['domain']]
        
        # 数据集信息
        dataset_info = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'num_classes': train_df['label'].nunique(),
            'vocab_size': len(char_to_idx),
            'max_length': max_length,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'class_distribution': train_df['label'].value_counts().sort_index().to_dict(),
            'family_distribution': train_df['family'].value_counts().to_dict(),
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'dataset_type': dataset_type
        }
        
        # 统一格式
        unified_dataset = {
            'info': dataset_info,
            'train': {
                'sequences': train_sequences,
                'labels': train_df['label'].tolist(),
                'families': train_df['family'].tolist()
            },
            'val': {
                'sequences': val_sequences,
                'labels': val_df['label'].tolist(),
                'families': val_df['family'].tolist()
            },
            'test': {
                'sequences': test_sequences,
                'labels': test_df['label'].tolist(),
                'families': test_df['family'].tolist()
            }
        }
        
        return unified_dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: str) -> None:
        """保存数据集"""
        print(f"保存数据集到 {output_path}...")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"数据集已保存: {output_path} ({file_size:.2f} MB)")
    
    def build_all_datasets(self) -> None:
        """构建所有数据集"""
        print("=" * 80)
        print("统一DGA数据集构建器")
        print("=" * 80)
        
        datasets_config = [
            # (样本数, 类型, 输出文件名)
            (10000, 'binary', 'small_binary_dga_dataset.pkl'),
            (100000, 'binary', 'large_binary_dga_dataset.pkl'),
            (10000, 'multiclass', 'small_multiclass_dga_dataset.pkl'),
            (100000, 'multiclass', 'large_multiclass_dga_dataset.pkl')
        ]
        
        for total_samples, dataset_type, filename in datasets_config:
            print(f"\n{'='*60}")
            print(f"构建 {dataset_type} 数据集 ({total_samples:,} 样本)")
            print(f"{'='*60}")
            
            # 构建数据集
            if dataset_type == 'binary':
                df = self.build_binary_dataset(total_samples)
            else:
                df = self.build_multiclass_dataset(total_samples)
            
            # 分割数据集
            train_df, val_df, test_df = self.split_dataset(df)
            
            # 创建统一格式
            unified_dataset = self.create_unified_format(train_df, val_df, test_df, dataset_type)
            
            # 保存数据集
            output_path = f'./data/processed/{filename}'
            self.save_dataset(unified_dataset, output_path)
            
            # 打印统计信息
            info = unified_dataset['info']
            print(f"\n📊 数据集统计:")
            print(f"  类型: {dataset_type}")
            print(f"  总样本: {info['total_samples']:,}")
            print(f"  训练集: {info['train_size']:,}")
            print(f"  验证集: {info['val_size']:,}")
            print(f"  测试集: {info['test_size']:,}")
            print(f"  类别数: {info['num_classes']}")
            print(f"  词汇表大小: {info['vocab_size']}")
            print(f"  类别分布: {info['class_distribution']}")


def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建构建器
    builder = UnifiedDGADatasetBuilder()
    
    # 构建所有数据集
    builder.build_all_datasets()
    
    print("\n" + "=" * 80)
    print("所有数据集构建完成！")
    print("=" * 80)
    print("\n生成的数据集:")
    print("1. small_binary_dga_dataset.pkl - 小型二分类数据集 (1万样本)")
    print("2. large_binary_dga_dataset.pkl - 大型二分类数据集 (10万样本)")
    print("3. small_multiclass_dga_dataset.pkl - 小型多分类数据集 (1万样本)")
    print("4. large_multiclass_dga_dataset.pkl - 大型多分类数据集 (10万样本)")
    print("\n所有数据集采用统一格式，支持8:1:1分割")


if __name__ == '__main__':
    main()