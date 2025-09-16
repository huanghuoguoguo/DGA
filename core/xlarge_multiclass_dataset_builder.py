#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超大规模多分类DGA数据集构建器
生成10万条数据，良性域名50%，恶意域名50%分布在多个DGA家族
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import string


class XLargeMulticlassDGADatasetBuilder:
    def __init__(self, total_samples=100000, benign_ratio=0.5):
        self.total_samples = total_samples
        self.benign_ratio = benign_ratio
        self.benign_samples = int(total_samples * benign_ratio)
        self.malicious_samples = total_samples - self.benign_samples
        
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
        
        # 验证权重总和
        total_weight = sum(family['weight'] for family in self.dga_families.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"警告: DGA家族权重总和为 {total_weight:.3f}，不等于1.0")
        
        # 常见TLD
        self.tlds = ['.com', '.net', '.org', '.info', '.biz', '.co', '.io', '.me', '.tv', '.cc']
        
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
    
    def generate_random_string(self, min_len, max_len, charset=None):
        """生成随机字符串"""
        if charset is None:
            charset = string.ascii_lowercase
        length = random.randint(min_len, max_len)
        return ''.join(random.choice(charset) for _ in range(length))
    
    def generate_dictionary_domain(self, min_len, max_len):
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
    
    def generate_mixed_domain(self, min_len, max_len):
        """生成混合型域名（字典+随机）"""
        # 随机选择1-2个字典单词
        words = random.sample(self.dictionary_words, random.randint(1, 2))
        base = ''.join(words)
        
        # 添加随机字符或数字
        remaining_len = random.randint(min_len - len(base), max_len - len(base))
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
    
    def generate_dga_domain(self, family_name, family_config):
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
    
    def generate_benign_domain(self):
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
    
    def build_dataset(self):
        """构建完整数据集"""
        print(f"开始构建超大规模多分类数据集...")
        print(f"总样本数: {self.total_samples:,}")
        print(f"良性域名: {self.benign_samples:,} ({self.benign_ratio*100:.1f}%)")
        print(f"恶意域名: {self.malicious_samples:,} ({(1-self.benign_ratio)*100:.1f}%)")
        print(f"DGA家族数: {len(self.dga_families)}")
        
        domains = []
        labels = []
        family_names = []
        
        # 生成良性域名
        print("\n生成良性域名...")
        for i in range(self.benign_samples):
            if (i + 1) % 10000 == 0:
                print(f"  已生成 {i+1:,} 个良性域名")
            
            domain = self.generate_benign_domain()
            domains.append(domain)
            labels.append(0)  # 良性标签为0
            family_names.append('benign')
        
        # 生成恶意域名
        print("\n生成恶意域名...")
        family_list = list(self.dga_families.keys())
        family_weights = [self.dga_families[f]['weight'] for f in family_list]
        
        # 根据权重分配每个家族的样本数
        family_counts = {}
        remaining_samples = self.malicious_samples
        
        for i, family in enumerate(family_list[:-1]):
            count = int(self.malicious_samples * family_weights[i])
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
            print(f"\n生成 {family} 家族域名 ({count:,} 个)...")
            family_config = self.dga_families[family]
            
            for i in range(count):
                if (i + 1) % 5000 == 0:
                    print(f"  已生成 {i+1:,} 个 {family} 域名")
                
                domain = self.generate_dga_domain(family, family_config)
                domains.append(domain)
                labels.append(label_counter)
                family_names.append(family)
            
            label_counter += 1
        
        # 创建DataFrame
        print("\n创建数据集...")
        df = pd.DataFrame({
            'domain': domains,
            'label': labels,
            'family': family_names
        })
        
        # 打乱数据
        print("打乱数据...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 统计信息
        print("\n数据集统计:")
        print(f"总样本数: {len(df):,}")
        print(f"类别数: {df['label'].nunique()}")
        print(f"家族分布:")
        family_dist = df['family'].value_counts().sort_index()
        for family, count in family_dist.items():
            percentage = count / len(df) * 100
            print(f"  {family}: {count:,} ({percentage:.2f}%)")
        
        # 域名长度统计
        df['domain_length'] = df['domain'].str.len()
        print(f"\n域名长度统计:")
        print(f"  平均长度: {df['domain_length'].mean():.2f}")
        print(f"  最短: {df['domain_length'].min()}")
        print(f"  最长: {df['domain_length'].max()}")
        print(f"  中位数: {df['domain_length'].median():.2f}")
        
        return df
    
    def split_dataset(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """分割数据集"""
        print(f"\n分割数据集 (训练:{train_ratio*100:.0f}%, 验证:{val_ratio*100:.0f}%, 测试:{test_ratio*100:.0f}%)...")
        
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
        
        # 验证分布
        print("\n各集合中的类别分布:")
        for name, dataset in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
            dist = dataset['label'].value_counts().sort_index()
            print(f"{name}: {dict(dist)}")
        
        return train_df, val_df, test_df
    
    def save_dataset(self, train_df, val_df, test_df, output_path):
        """保存数据集"""
        print(f"\n保存数据集到 {output_path}...")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 准备数据
        dataset_info = {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'num_classes': train_df['label'].nunique(),
            'class_distribution': train_df['label'].value_counts().sort_index().to_dict(),
            'family_distribution': train_df['family'].value_counts().to_dict(),
            'vocab_size': None,  # 将在处理时计算
            'max_length': 40,  # 固定最大长度
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
        
        # 字符到索引的映射
        all_chars = set()
        for domain in train_df['domain']:
            all_chars.update(domain.lower())
        
        # 添加特殊字符
        all_chars.add('<PAD>')  # 填充
        all_chars.add('<UNK>')  # 未知字符
        
        char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        dataset_info['vocab_size'] = len(char_to_idx)
        dataset_info['char_to_idx'] = char_to_idx
        dataset_info['idx_to_char'] = idx_to_char
        
        # 转换域名为数字序列
        def domain_to_sequence(domain, max_len=40):
            sequence = [char_to_idx.get(char.lower(), char_to_idx['<UNK>']) for char in domain]
            # 截断或填充
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            else:
                sequence.extend([char_to_idx['<PAD>']] * (max_len - len(sequence)))
            return sequence
        
        # 处理数据
        train_data = {
            'sequences': [domain_to_sequence(domain) for domain in train_df['domain']],
            'labels': train_df['label'].tolist(),
            'families': train_df['family'].tolist()
        }
        
        val_data = {
            'sequences': [domain_to_sequence(domain) for domain in val_df['domain']],
            'labels': val_df['label'].tolist(),
            'families': val_df['family'].tolist()
        }
        
        test_data = {
            'sequences': [domain_to_sequence(domain) for domain in test_df['domain']],
            'labels': test_df['label'].tolist(),
            'families': test_df['family'].tolist()
        }
        
        # 保存数据
        final_dataset = {
            'info': dataset_info,
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_dataset, f)
        
        print(f"数据集已保存: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return dataset_info


def main():
    """主函数"""
    print("=" * 60)
    print("超大规模多分类DGA数据集构建器")
    print("=" * 60)
    
    # 创建构建器 - 总样本数约14.3万，确保训练集达到10万
    builder = XLargeMulticlassDGADatasetBuilder(
        total_samples=143000,  # 约14.3万样本，训练集70%约10万
        benign_ratio=0.5       # 50%良性
    )
    
    # 构建数据集
    df = builder.build_dataset()
    
    # 分割数据集
    train_df, val_df, test_df = builder.split_dataset(df)
    
    # 保存数据集
    output_path = './data/processed/xlarge_multiclass_dga_dataset.pkl'
    dataset_info = builder.save_dataset(train_df, val_df, test_df, output_path)
    
    print("\n=" * 60)
    print("数据集构建完成！")
    print("=" * 60)
    print(f"数据集路径: {output_path}")
    print(f"总样本数: {dataset_info['total_samples']:,}")
    print(f"类别数: {dataset_info['num_classes']}")
    print(f"词汇表大小: {dataset_info['vocab_size']}")
    print(f"训练集: {dataset_info['train_size']:,}")
    print(f"验证集: {dataset_info['val_size']:,}")
    print(f"测试集: {dataset_info['test_size']:,}")


if __name__ == '__main__':
    main()