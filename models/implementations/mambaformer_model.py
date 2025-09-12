#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - MambaFormer模型实现
结合Mamba状态空间模型和Transformer的混合架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


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


class MambaBlock(nn.Module):
    """Mamba状态空间模型块"""
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super(MambaBlock, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # 选择性扫描参数
        self.input_proj = nn.Linear(d_model, d_state)
        self.state_proj = nn.Linear(d_model, d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        
        # 门控机制
        self.gate = nn.Linear(d_model, d_model)
        self.gate_activation = nn.Sigmoid()
        
        # Delta参数（时间步长控制）
        self.delta_proj = nn.Linear(d_model, d_state)
        
        # 状态转移矩阵（可学习）
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        
        # 归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        x = self.norm(x)
        
        # 选择性扫描机制
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)
            
            # 计算选择性参数
            delta = torch.sigmoid(self.delta_proj(x_t))  # (batch, d_state)
            B_t = self.input_proj(x_t)  # (batch, d_state)
            
            # 状态更新（简化的选择性扫描）
            A_scaled = self.A * delta.unsqueeze(-1)  # (batch, d_state, d_state)
            h = torch.tanh(torch.matmul(h.unsqueeze(1), A_scaled).squeeze(1) + B_t)
            
            # 输出计算
            y_t = torch.matmul(h.unsqueeze(1), self.C).squeeze(1)  # (batch, d_model)
            
            # 门控
            gate_t = self.gate_activation(self.gate(x_t))
            y_t = y_t * gate_t
            
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        # 残差连接
        return self.dropout(output) + residual


class TransformerBlock(nn.Module):
    """标准Transformer块"""
    
    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: attention mask
        """
        # 自注意力 + 残差连接
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # 前馈网络 + 残差连接
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x


class MambaFormerBlock(nn.Module):
    """MambaFormer混合块"""
    
    def __init__(self, d_model, d_state=16, n_heads=8, dropout=0.1, fusion_type='parallel'):
        super(MambaFormerBlock, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Mamba和Transformer组件
        self.mamba_block = MambaBlock(d_model, d_state, dropout)
        self.transformer_block = TransformerBlock(d_model, n_heads, dropout=dropout)
        
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
            # 序列化：先Mamba后Transformer
            x = self.mamba_block(x)
            x = self.transformer_block(x, mask)
            return x
            
        elif self.fusion_type == 'parallel':
            # 并行：两个分支并行处理后融合
            mamba_out = self.mamba_block(x)
            transformer_out = self.transformer_block(x, mask)
            
            # 拼接并融合
            combined = torch.cat([mamba_out, transformer_out], dim=-1)
            fused = self.fusion(combined)
            
            return fused
            
        elif self.fusion_type == 'gated':
            # 门控融合：动态权重组合
            mamba_out = self.mamba_block(x)
            transformer_out = self.transformer_block(x, mask)
            
            # 计算门控权重
            combined = torch.cat([mamba_out, transformer_out], dim=-1)
            weights = self.gate(combined)  # (batch, seq_len, 2)
            
            # 加权融合
            fused = (weights[:, :, 0:1] * mamba_out + 
                    weights[:, :, 1:2] * transformer_out)
            
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