#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - BiGRU+Attention模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class BiGRUAttentionModel(BaseModel):
    """BiGRU+Attention模型用于DGA检测"""
    
    def __init__(self, vocab_size, d_model=128, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.1):
        super(BiGRUAttentionModel, self).__init__(vocab_size, num_classes)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # BiGRU层
        self.gru = nn.GRU(
            d_model, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # 注意力机制
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 嵌入 (batch, seq_len, d_model)
        x = self.embedding(x)
        x = self.dropout(x)
        
        # BiGRU (batch, seq_len, hidden_size*2)
        gru_out, _ = self.gru(x)
        
        # 注意力机制
        # 计算注意力权重 (batch, seq_len, 1)
        attention_weights = torch.tanh(self.attention(gru_out))
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和 (batch, hidden_size*2)
        context = torch.sum(gru_out * attention_weights, dim=1)
        
        # 分类
        output = self.classifier(context)
        
        return output
    
    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"BiGRU+Attention Model Info:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")