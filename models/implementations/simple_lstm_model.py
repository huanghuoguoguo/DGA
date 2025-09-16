#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 简单LSTM模型实现
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class SimpleLSTMModel(BaseModel):
    """简单LSTM模型用于DGA检测"""
    
    def __init__(self, vocab_size, d_model=128, hidden_size=128, num_layers=1, 
                 num_classes=2, dropout=0.1):
        super(SimpleLSTMModel, self).__init__(vocab_size, num_classes)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 单向LSTM层
        self.lstm = nn.LSTM(
            d_model, hidden_size, num_layers,
            batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0
        )
        
        # 简单分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 嵌入 (batch, seq_len, d_model)
        x = self.embedding(x)
        x = self.dropout(x)
        
        # LSTM (batch, seq_len, hidden_size)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # 使用最后一个时间步的隐藏状态
        # hidden: (num_layers, batch, hidden_size)
        last_hidden = hidden[-1]  # (batch, hidden_size)
        
        # 分类
        logits = self.classifier(last_hidden)
        return logits