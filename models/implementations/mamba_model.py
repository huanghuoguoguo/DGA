#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - Mamba模型实现（简化版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class MambaBlock(nn.Module):
    """简化的Mamba块 - 使用RNN模拟状态空间模型"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # 状态空间参数
        self.state_proj = nn.Linear(d_model, d_state)
        self.input_proj = nn.Linear(d_model, d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        
        # 门控机制
        self.gate = nn.Linear(d_model, d_model)
        
        # 归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 状态转移矩阵
        self.state_transition = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            
            # 状态更新
            h_proj = self.state_proj(x_t)  # (batch, d_state)
            x_proj = self.input_proj(x_t)  # (batch, d_state)
            
            # 简化的状态空间更新
            h = torch.tanh(h @ self.state_transition.T + h_proj + x_proj)
            
            # 输出投影
            output_t = self.output_proj(h)  # (batch, d_model)
            
            # 门控
            gate_t = torch.sigmoid(self.gate(x_t))
            output_t = output_t * gate_t
            
            outputs.append(output_t)
        
        # 拼接所有时间步的输出
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        # 残差连接和dropout
        output = self.dropout(output) + residual
        
        return output


class MambaModel(BaseModel):
    """简化的Mamba模型用于DGA检测"""
    
    def __init__(self, vocab_size, d_model=128, n_layers=3, d_state=16, 
                 num_classes=2, dropout=0.1):
        super(MambaModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x):
        # 嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # Mamba层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)
        
        return logits