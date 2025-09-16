#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构混合专家模型(Heterogeneous MoE)
使用不同架构的专家网络，每个专家专注于不同类型的特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from typing import Tuple, Optional, List


class CNNExpert(nn.Module):
    """CNN专家 - 专注于局部模式识别"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, 64, kernel_size=k, padding=k//2)
            for k in [2, 3, 4, 5]
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(64) for _ in range(4)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(64 * 4, d_model)
        
        # 专家特化层
        self.expert_norm = nn.LayerNorm(d_model)
        self.expert_gate = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # 嵌入
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # 多尺度卷积
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        # 特征融合
        features = torch.cat(conv_outputs, dim=1)
        output = self.output_proj(features)
        
        # 专家特化
        output = self.expert_norm(output)
        gate = torch.sigmoid(self.expert_gate(output))
        output = output * gate
        
        return output


class LSTMExpert(nn.Module):
    """LSTM专家 - 专注于序列依赖建模"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            d_model, d_model // 2, 
            batch_first=True, 
            bidirectional=True,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 专家特化层
        self.expert_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x):
        # 嵌入
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm1(lstm_out + x)
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm2(attn_out + lstm_out)
        
        # 全局平均池化
        mask = (x.sum(dim=-1) != 0).float().unsqueeze(-1)  # padding mask
        pooled = (attn_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 专家特化
        output = self.expert_ffn(pooled)
        
        return output


class TransformerExpert(nn.Module):
    """Transformer专家 - 专注于全局上下文建模"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_encoding = self._create_pos_encoding(100, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # 专家特化层
        self.expert_proj = nn.Linear(d_model, d_model)
        
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # 嵌入和位置编码
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = embedded + self.pos_encoding[:, :x.size(1), :].to(x.device)
        embedded = self.dropout(embedded)
        
        # 创建padding mask
        padding_mask = (x == 0)
        
        # Transformer处理
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # 全局平均池化（忽略padding）
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 专家特化
        output = self.expert_proj(self.norm(pooled))
        
        return output


class MambaExpert(nn.Module):
    """Mamba专家 - 专注于长序列高效建模"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 状态空间参数
        self.state_size = d_model // 4
        self.expand_factor = 2
        
        # 输入投影
        self.input_proj = nn.Linear(d_model, d_model * self.expand_factor)
        
        # 状态空间层
        self.conv1d = nn.Conv1d(
            d_model * self.expand_factor, 
            d_model * self.expand_factor, 
            kernel_size=3, 
            padding=1, 
            groups=d_model * self.expand_factor
        )
        
        # 选择性机制
        self.x_proj = nn.Linear(d_model * self.expand_factor, self.state_size)
        self.dt_proj = nn.Linear(d_model * self.expand_factor, d_model * self.expand_factor)
        
        # 状态转移矩阵
        self.A_log = nn.Parameter(torch.randn(d_model * self.expand_factor, self.state_size))
        self.D = nn.Parameter(torch.randn(d_model * self.expand_factor))
        
        # 输出投影
        self.output_proj = nn.Linear(d_model * self.expand_factor, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, d_model)
        embedded = self.dropout(embedded)
        
        # 输入投影
        x_proj = self.input_proj(embedded)  # (batch, seq_len, d_model * expand_factor)
        
        # 卷积处理
        x_conv = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        x_processed = F.silu(x_conv)
        
        # 选择性状态空间
        dt = F.softplus(self.dt_proj(x_processed))  # (batch, seq_len, d_model * expand_factor)
        A = -torch.exp(self.A_log.float())  # (d_model * expand_factor, state_size)
        
        # 简化的状态空间计算
        x_state = self.x_proj(x_processed)  # (batch, seq_len, state_size)
        
        # 状态更新（简化版）
        h = torch.zeros(batch_size, self.state_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            dt_t = dt[:, t, :self.state_size]  # 取前state_size维
            x_t = x_state[:, t, :]
            
            # 状态更新
            h = h * torch.exp(A[:self.state_size, :].mean(0) * dt_t.mean(-1, keepdim=True)) + x_t
            
            # 输出计算 - 简化为线性变换
            y_t = h  # 直接使用状态作为输出
            outputs.append(y_t)
        
        # 组合输出
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, state_size)
        
        # 扩展到d_model * expand_factor维度
        d_model_expand = embedded.size(-1) * self.expand_factor
        if output.size(-1) < d_model_expand:
            # 重复扩展到目标维度
            repeat_factor = d_model_expand // output.size(-1)
            remainder = d_model_expand % output.size(-1)
            output_expanded = output.repeat(1, 1, repeat_factor)
            if remainder > 0:
                output_expanded = torch.cat([output_expanded, output[:, :, :remainder]], dim=-1)
        else:
            output_expanded = output[:, :, :d_model_expand]
        
        # 残差连接
        output_with_residual = output_expanded + x_processed * self.D.unsqueeze(0).unsqueeze(0)
        
        # 全局平均池化并投影到d_model
        pooled = output_with_residual.mean(dim=1)  # (batch, d_model * expand_factor)
        output_final = self.output_proj(pooled)  # (batch, d_model)
        
        return self.norm(output_final)


class HeterogeneousGatingNetwork(nn.Module):
    """异构门控网络 - 为不同专家设计不同的门控策略"""
    
    def __init__(self, vocab_size: int, d_model: int, num_experts: int = 4, 
                 dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        
        self.num_experts = num_experts
        self.temperature = temperature
        
        # 输入嵌入
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 多层门控网络
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        # 专家特定的门控头
        self.expert_gates = nn.ModuleList([
            nn.Linear(d_model // 4, 1) for _ in range(num_experts)
        ])
        
        # 全局门控
        self.global_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_experts)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 负载均衡
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def forward(self, x, training=True):
        batch_size, seq_len = x.shape
        
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, d_model)
        embedded = self.dropout(embedded)
        
        # 全局平均池化
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (batch, d_model)
        
        # 计算每个专家的门控分数
        expert_scores = []
        for i, (gate_layer, expert_gate) in enumerate(zip(self.gate_layers, self.expert_gates)):
            gate_features = gate_layer(pooled)  # (batch, d_model//4)
            score = expert_gate(gate_features)  # (batch, 1)
            expert_scores.append(score)
        
        expert_scores = torch.cat(expert_scores, dim=-1)  # (batch, num_experts)
        
        # 全局门控调节
        global_scores = self.global_gate(pooled)  # (batch, num_experts)
        
        # 组合门控分数
        combined_scores = (expert_scores + global_scores) / self.temperature
        
        # Softmax归一化
        gate_weights = F.softmax(combined_scores, dim=-1)
        
        # 负载均衡损失
        load_loss = self._compute_load_balancing_loss(gate_weights)
        
        # 更新统计
        if training:
            self._update_expert_stats(gate_weights)
        
        return gate_weights, load_loss
    
    def _compute_load_balancing_loss(self, gate_weights):
        """计算负载均衡损失"""
        # 计算每个专家的平均使用率
        expert_usage = gate_weights.mean(dim=0)  # (num_experts,)
        
        # 理想使用率
        target_usage = 1.0 / self.num_experts
        
        # 计算方差作为负载均衡损失
        load_loss = torch.var(expert_usage) * self.num_experts
        
        return load_loss
    
    def _update_expert_stats(self, gate_weights):
        """更新专家使用统计"""
        batch_usage = gate_weights.sum(dim=0)
        self.expert_counts += batch_usage
        self.total_tokens += gate_weights.size(0)
    
    def get_expert_utilization(self):
        """获取专家利用率统计"""
        if self.total_tokens > 0:
            utilization = self.expert_counts / self.total_tokens
            return {
                'expert_counts': self.expert_counts.cpu().numpy(),
                'utilization': utilization.cpu().numpy(),
                'total_tokens': self.total_tokens.item(),
                'balance_score': 1.0 - utilization.std().item()
            }
        return None


class HeterogeneousMoEModel(BaseModel):
    """异构MoE模型 - 使用不同架构的专家网络"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_classes: int = 2, 
                 dropout: float = 0.1, load_balance_weight: float = 0.01,
                 temperature: float = 1.0):
        super(HeterogeneousMoEModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.num_experts = 4
        self.load_balance_weight = load_balance_weight
        
        # 创建异构专家
        self.experts = nn.ModuleList([
            CNNExpert(vocab_size, d_model, dropout),      # 专家0: CNN
            LSTMExpert(vocab_size, d_model, dropout),     # 专家1: LSTM
            TransformerExpert(vocab_size, d_model, dropout), # 专家2: Transformer
            MambaExpert(vocab_size, d_model, dropout)     # 专家3: Mamba
        ])
        
        # 异构门控网络
        self.gate = HeterogeneousGatingNetwork(
            vocab_size, d_model, self.num_experts, dropout, temperature
        )
        
        # 专家输出融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 门控网络计算专家权重
        gate_weights, load_loss = self.gate(x, self.training)
        
        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)  # (batch, d_model)
            expert_outputs.append(output)
        
        # 加权融合专家输出
        mixed_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            weight = gate_weights[:, i:i+1]  # (batch, 1)
            mixed_output += weight * expert_output
        
        # 融合层处理
        fused_output = self.fusion_layer(mixed_output)
        
        # 分类
        logits = self.classifier(fused_output)
        
        # 只返回logits，将load_loss存储为模型属性
        self.last_load_loss = load_loss * self.load_balance_weight
        return logits
    
    def get_expert_utilization(self):
        """获取专家利用率统计"""
        return self.gate.get_expert_utilization()
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        expert_info = {
            'expert_0': 'CNN (局部模式)',
            'expert_1': 'LSTM (序列依赖)', 
            'expert_2': 'Transformer (全局上下文)',
            'expert_3': 'Mamba (长序列建模)'
        }
        
        return {
            'model_name': 'HeterogeneousMoE',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'd_model': self.d_model,
            'num_experts': self.num_experts,
            'expert_types': expert_info,
            'architecture': 'Heterogeneous (CNN + LSTM + Transformer + Mamba)'
        }