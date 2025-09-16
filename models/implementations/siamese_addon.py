#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孪生网络附加模块
用于增强DGA检测模型的小样本学习能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SiameseHead(nn.Module):
    """轻量级孪生头，输出归一化向量"""
    
    def __init__(self, in_dim=256, emb_dim=128, dropout=0.1):
        super(SiameseHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, x):
        """输出L2归一化的嵌入向量，方便计算cosine相似度"""
        return F.normalize(self.net(x), p=2, dim=1)


class CosineContrastiveLoss(nn.Module):
    """余弦对比损失函数"""
    
    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, z1, z2, same_family):
        """
        Args:
            z1, z2: 归一化的嵌入向量 [batch_size, emb_dim]
            same_family: 是否同家族标签 [batch_size] (1: 同家族, 0: 不同家族)
        """
        cos_sim = F.cosine_similarity(z1, z2)  # [-1, 1]
        
        # 同家族：拉近距离 (最大化cosine相似度)
        # 不同家族：推开距离 (最小化cosine相似度，但不超过margin)
        loss = torch.where(
            same_family == 1,
            1 - cos_sim,  # 同家族：损失 = 1 - cos_sim
            F.relu(cos_sim - self.margin)  # 不同家族：只有当cos_sim > margin时才有损失
        )
        
        return loss.mean()


class TripletLoss(nn.Module):
    """三元组损失函数"""
    
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: 锚点样本嵌入 [batch_size, emb_dim]
            positive: 正样本嵌入 [batch_size, emb_dim] 
            negative: 负样本嵌入 [batch_size, emb_dim]
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class SiameseDataset(torch.utils.data.Dataset):
    """孪生网络数据集包装器"""
    
    def __init__(self, base_dataset, positive_ratio=0.6):
        """
        Args:
            base_dataset: 原始数据集
            positive_ratio: 正样本对的比例
        """
        self.base_dataset = base_dataset
        self.positive_ratio = positive_ratio
        
        # 按类别组织数据索引
        self.class_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            # 确保标签是Python标量
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 获取锚点样本
        anchor_data, anchor_label = self.base_dataset[idx]
        
        # 确保标签是Python标量
        if isinstance(anchor_label, torch.Tensor):
            anchor_label = anchor_label.item()
        
        # 决定生成正样本对还是负样本对
        if torch.rand(1).item() < self.positive_ratio:
            # 生成正样本对（同类别）
            positive_indices = self.class_indices[anchor_label]
            # 排除自己
            positive_indices = [i for i in positive_indices if i != idx]
            if positive_indices:
                pos_idx = torch.randint(0, len(positive_indices), (1,)).item()
                pair_idx = positive_indices[pos_idx]
            else:
                pair_idx = idx  # 如果没有其他同类样本，使用自己
            same_family = 1
        else:
            # 生成负样本对（不同类别）
            other_classes = [c for c in self.classes if c != anchor_label]
            if other_classes:
                neg_class = torch.randint(0, len(other_classes), (1,)).item()
                neg_class_label = other_classes[neg_class]
                neg_indices = self.class_indices[neg_class_label]
                neg_idx = torch.randint(0, len(neg_indices), (1,)).item()
                pair_idx = neg_indices[neg_idx]
            else:
                pair_idx = idx  # 如果只有一个类别，使用自己
            same_family = 0
            
        pair_data, pair_label = self.base_dataset[pair_idx]
        
        return {
            'anchor': anchor_data,
            'pair': pair_data,
            'same_family': same_family,
            'anchor_label': anchor_label,
            'pair_label': pair_label
        }


def create_siamese_dataloader(base_dataset, batch_size=32, positive_ratio=0.6, 
                             shuffle=True, num_workers=0):
    """创建孪生网络数据加载器"""
    siamese_dataset = SiameseDataset(base_dataset, positive_ratio)
    return torch.utils.data.DataLoader(
        siamese_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )