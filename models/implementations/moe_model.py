#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - MoE模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from .cnn_model import CNNModel
from .lstm_model import LSTMModel


class MoEModel(BaseModel):
    """混合专家模型（CNN + LSTM）"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(MoEModel, self).__init__(vocab_size, num_classes)
        
        # 专家网络（移除最后的分类层）
        self.cnn_expert = self._create_cnn_expert(vocab_size, d_model, dropout)
        self.lstm_expert = self._create_lstm_expert(vocab_size, d_model, dropout)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Embedding(vocab_size, d_model, padding_idx=0),
            nn.LSTM(d_model, 64, batch_first=True),
            nn.Linear(64, 2),  # 2个专家
            nn.Softmax(dim=-1)
        )
        
        # 最终分类器
        self.classifier = nn.Linear(d_model, num_classes)
        
    def _create_cnn_expert(self, vocab_size, d_model, dropout):
        """创建CNN专家（去掉分类层）"""
        class CNNExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.conv_layers = nn.ModuleList([
                    nn.Conv1d(d_model, 128, kernel_size=k, padding=k//2)
                    for k in [2, 3, 4]
                ])
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(128) for _ in range(3)
                ])
                self.dropout = nn.Dropout(dropout)
                self.output_proj = nn.Linear(128 * 3, d_model)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.dropout(x)
                x = x.transpose(1, 2)
                
                conv_outputs = []
                for conv, bn in zip(self.conv_layers, self.batch_norms):
                    conv_out = F.relu(bn(conv(x)))
                    pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
                    conv_outputs.append(pooled.squeeze(2))
                
                features = torch.cat(conv_outputs, dim=1)
                return self.output_proj(features)
        
        return CNNExpert()
    
    def _create_lstm_expert(self, vocab_size, d_model, dropout):
        """创建LSTM专家（去掉分类层）"""
        class LSTMExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.lstm = nn.LSTM(d_model, 64, batch_first=True, bidirectional=True)
                self.attention = nn.Linear(128, 1)
                self.dropout = nn.Dropout(dropout)
                self.output_proj = nn.Linear(128, d_model)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.dropout(x)
                
                lstm_out, _ = self.lstm(x)
                attention_weights = F.softmax(self.attention(lstm_out), dim=1)
                weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
                
                return self.output_proj(weighted_output)
        
        return LSTMExpert()
    
    def forward(self, x):
        # 门控网络计算专家权重
        gate_input = x
        gate_out, (h_n, _) = self.gate[1](self.gate[0](gate_input))
        gate_weights = self.gate[2](h_n[-1])  # (batch, 2)
        
        # 专家输出
        cnn_output = self.cnn_expert(x)  # (batch, d_model)
        lstm_output = self.lstm_expert(x)  # (batch, d_model)
        
        # 加权混合
        mixed_output = (gate_weights[:, 0:1] * cnn_output + 
                       gate_weights[:, 1:2] * lstm_output)
        
        # 最终分类
        logits = self.classifier(mixed_output)
        
        return logits