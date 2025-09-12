#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - CNN模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class CNNModel(BaseModel):
    """CNN模型用于DGA检测"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(CNNModel, self).__init__(vocab_size, num_classes)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, 128, kernel_size=k, padding=k//2)
            for k in [2, 3, 4]
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(128) for _ in range(3)
        ])
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 嵌入 (batch, seq_len, d_model)
        x = self.embedding(x)
        x = self.dropout(x)
        
        # 转置为 (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))  # (batch, 128, seq_len)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch, 128, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch, 128)
        
        # 拼接特征
        features = torch.cat(conv_outputs, dim=1)  # (batch, 128*3)
        
        # 分类
        logits = self.classifier(features)
        return logits