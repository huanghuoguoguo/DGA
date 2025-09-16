#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - Mamba模型实现（使用zeta提供的MambaBlock）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from zeta.nn import MambaBlock


class MambaWrapper(nn.Module):
    """使用zeta提供的MambaBlock的包装器"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(MambaWrapper, self).__init__()
        
        # 使用zeta提供的MambaBlock
        self.mamba_block = MambaBlock(
            dim=d_model,
            depth=1,  # 单层
            d_state=d_state,
            expand=expand,
            d_conv=d_conv
        )
        
        # 归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 使用zeta的MambaBlock
        output = self.mamba_block(x)
        
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
            MambaWrapper(d_model, d_state, dropout=dropout)
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