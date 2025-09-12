#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - LSTM模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class LSTMModel(BaseModel):
    """BiLSTM+Attention模型用于DGA检测"""
    
    def __init__(self, vocab_size, d_model=128, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.1):
        super(LSTMModel, self).__init__(vocab_size, num_classes)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
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
        
        # BiLSTM (batch, seq_len, hidden_size*2)
        lstm_out, _ = self.lstm(x)
        
        # 注意力权重 (batch, seq_len, 1)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # 加权求和 (batch, hidden_size*2)
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # 分类
        logits = self.classifier(weighted_output)
        return logits