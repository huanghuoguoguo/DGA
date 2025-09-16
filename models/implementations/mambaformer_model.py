#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - MambaFormer模型实现
结合Mamba状态空间模型和Transformer的混合架构（使用zeta实现）
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
from zeta.nn.attention.multiquery_attention import MultiQueryAttention


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_length=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class MambaWrapper(nn.Module):
    """使用zeta提供的MambaBlock的包装器"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super(MambaWrapper, self).__init__()
        
        # 使用zeta提供的MambaBlock
        self.mamba_block = MambaBlock(
            dim=d_model,
            depth=1,  # 单层
            d_state=d_state,
            expand=2,
            d_conv=4
        )
        
        # 归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 使用zeta的MambaBlock
        output = self.mamba_block(x)
        
        # 残差连接
        return self.dropout(output) + residual


class MambaFormerBlock(nn.Module):
    """MambaFormer混合块（使用zeta实现）"""
    
    def __init__(self, d_model, d_state=16, n_heads=8, dropout=0.1, fusion_type='parallel'):
        super(MambaFormerBlock, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Mamba和Transformer组件
        self.mamba_block = MambaWrapper(d_model, d_state, dropout)
        self.attention_block = MultiQueryAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        if fusion_type == 'parallel':
            # 并行融合：需要fusion层
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'gated':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        """
        if self.fusion_type == 'sequential':
            # 序列化：先Mamba后Attention
            x = self.mamba_block(x)
            # 使用MultiQueryAttention
            x_norm = self.norm1(x)
            attn_out, _, _ = self.attention_block(x_norm)
            x = x + self.dropout_layer(attn_out)
            return x
            
        elif self.fusion_type == 'parallel':
            # 并行：两个分支并行处理后融合
            mamba_out = self.mamba_block(x)
            
            # Attention分支
            x_norm = self.norm1(x)
            attn_out, _, _ = self.attention_block(x_norm)
            attention_out = x + self.dropout_layer(attn_out)
            
            # 拼接并融合
            combined = torch.cat([mamba_out, attention_out], dim=-1)
            fused = self.fusion(combined)
            
            return fused
            
        elif self.fusion_type == 'gated':
            # 门控融合：动态权重组合
            mamba_out = self.mamba_block(x)
            
            # Attention分支
            x_norm = self.norm1(x)
            attn_out, _, _ = self.attention_block(x_norm)
            attention_out = x + self.dropout_layer(attn_out)
            
            # 计算门控权重
            combined = torch.cat([mamba_out, attention_out], dim=-1)
            weights = self.gate(combined)  # (batch, seq_len, 2)
            
            # 加权融合
            fused = (weights[:, :, 0:1] * mamba_out + 
                    weights[:, :, 1:2] * attention_out)
            
            return fused
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")


class MambaFormerModel(BaseModel):
    """MambaFormer模型用于DGA检测"""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16, 
                 n_heads=8, num_classes=2, dropout=0.1, fusion_type='gated'):
        super(MambaFormerModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # MambaFormer层
        self.layers = nn.ModuleList([
            MambaFormerBlock(d_model, d_state, n_heads, dropout, fusion_type)
            for _ in range(n_layers)
        ])
        
        # 最终归一化
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
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len)
        mask: attention mask (optional)
        """
        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # MambaFormer层
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def get_fusion_info(self):
        """获取融合策略信息"""
        return {
            'fusion_type': self.fusion_type,
            'n_layers': len(self.layers),
            'd_model': self.d_model,
            'architecture': 'MambaFormer'
        }