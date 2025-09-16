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
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            d_model, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 改进的分类器
        lstm_output_size = hidden_size * 2  # 双向LSTM
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 改进的权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
            elif 'weight' in name and len(param.shape) == 2:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        # 创建mask处理padding
        mask = (x != 0).float()
        lengths = mask.sum(dim=1).long()
        
        # 嵌入 (batch, seq_len, d_model)
        x = self.embedding(x)
        x = self.dropout(x)
        
        # 打包序列处理不同长度
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM (batch, seq_len, hidden_size*2)
        packed_out, (hidden, _) = self.lstm(packed_x)
        
        # 解包
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # 使用最后一个有效时间步的输出
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_outputs.append(lstm_out[i, lengths[i]-1, :])
        
        last_output = torch.stack(last_outputs)
        
        # 分类
        logits = self.classifier(last_output)
        return logits