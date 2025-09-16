#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的同构混合专家模型(Improved Homogeneous MoE)
使用相同架构但具有更强专家特化能力的专家网络
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


class ImprovedHomogeneousExpert(nn.Module):
    """改进的同构专家网络 - 增强专家特化能力"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, expert_id: int = 0):
        super().__init__()
        
        self.expert_id = expert_id
        self.d_model = d_model
        
        # 专家特化的输入变换
        self.input_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 主要的专家网络（多层）
        self.expert_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model if i == 0 else d_ff, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_ff)
            ) for i in range(2)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(d_ff, d_model)
        
        # 专家特化参数（每个专家有不同的特化方向）
        self.expert_bias = nn.Parameter(torch.randn(d_model) * 0.02)
        self.expert_scale = nn.Parameter(torch.ones(d_model) + torch.randn(d_model) * 0.01)
        
        # 专家门控（内部特化）
        self.internal_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        
        # 根据专家ID初始化不同的特化方向
        self._init_expert_specialization()
        
    def _init_expert_specialization(self):
        """根据专家ID初始化不同的特化方向"""
        # 为不同专家设置不同的初始化偏好
        with torch.no_grad():
            if self.expert_id % 4 == 0:  # 专家0: 偏向低频特征
                self.expert_bias.data *= 0.5
            elif self.expert_id % 4 == 1:  # 专家1: 偏向高频特征
                self.expert_bias.data *= 1.5
            elif self.expert_id % 4 == 2:  # 专家2: 偏向中等特征
                self.expert_bias.data *= 1.0
            else:  # 专家3: 偏向稀疏特征
                self.expert_scale.data *= 0.8
                
    def forward(self, x):
        # 输入变换
        transformed = self.input_transform(x)
        
        # 通过专家层
        hidden = transformed
        for layer in self.expert_layers:
            hidden = layer(hidden)
        
        # 输出投影
        output = self.output_proj(hidden)
        
        # 内部门控
        gate = self.internal_gate(x)
        output = output * gate
        
        # 专家特化
        output = output * self.expert_scale + self.expert_bias
        
        # 残差连接
        output = self.residual_weight * output + (1 - self.residual_weight) * x
        
        return self.dropout(output)


class AdaptiveGatingNetwork(nn.Module):
    """自适应门控网络 - 动态调整门控策略"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, 
                 dropout: float = 0.1, noise_std: float = 0.1, 
                 temperature: float = 1.0, adaptive_temp: bool = True):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.temperature = temperature
        self.adaptive_temp = adaptive_temp
        
        # 多层门控网络
        self.gate_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_experts)
        )
        
        # 自适应温度网络
        if adaptive_temp:
            self.temp_network = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Softplus()
            )
        
        # 上下文感知门控
        self.context_gate = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # 负载均衡相关
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        self.register_buffer('expert_variance_history', torch.zeros(100))  # 记录方差历史
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
        
        # 动态负载均衡权重
        self.register_buffer('dynamic_balance_weight', torch.tensor(1.0))
        
    def forward(self, x, training=True):
        """
        x: (batch_size, seq_len, d_model)
        返回: (gates, indices, load_balancing_loss)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 上下文感知处理
        context_x, _ = self.context_gate(x, x, x)
        context_x = context_x + x  # 残差连接
        
        # 全局平均池化用于门控
        pooled = context_x.mean(dim=1)  # (batch_size, d_model)
        
        # 计算门控分数
        gate_logits = self.gate_layers(pooled)  # (batch_size, num_experts)
        
        # 自适应温度
        if self.adaptive_temp:
            temp = self.temp_network(pooled).squeeze(-1)  # (batch_size,)
            temp = torch.clamp(temp, min=0.1, max=5.0)
        else:
            temp = self.temperature
        
        # 应用温度
        if isinstance(temp, torch.Tensor):
            gate_logits = gate_logits / temp.unsqueeze(-1)
        else:
            gate_logits = gate_logits / temp
        
        # 添加噪声（训练时）
        if training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k选择
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # 计算门控权重
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # 负载均衡损失
        load_balancing_loss = self._compute_adaptive_load_balancing_loss(gate_logits)
        
        # 更新专家使用统计
        if training:
            self._update_expert_stats(top_k_indices)
            self._update_balance_weight(gate_logits)
        
        return top_k_gates, top_k_indices, load_balancing_loss
    
    def _compute_adaptive_load_balancing_loss(self, gate_logits):
        """计算自适应负载均衡损失"""
        # 计算每个专家的平均概率
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_probs = gate_probs.mean(dim=0)  # (num_experts,)
        
        # 理想概率
        target_prob = 1.0 / self.num_experts
        
        # 计算方差损失
        variance_loss = torch.var(expert_probs) * self.num_experts
        
        # 计算熵损失（鼓励多样性）
        entropy_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8))
        max_entropy = math.log(self.num_experts)
        entropy_loss = max_entropy - entropy_loss
        
        # 组合损失
        total_loss = variance_loss + 0.1 * entropy_loss
        
        return total_loss * self.dynamic_balance_weight
    
    def _update_expert_stats(self, top_k_indices):
        """更新专家使用统计"""
        batch_size = top_k_indices.size(0)
        
        # 统计每个专家的使用次数
        for expert_idx in range(self.num_experts):
            count = (top_k_indices == expert_idx).sum().float()
            self.expert_counts[expert_idx] += count
        
        self.total_tokens += batch_size
    
    def _update_balance_weight(self, gate_logits):
        """动态更新负载均衡权重"""
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_probs = gate_probs.mean(dim=0)
        current_variance = torch.var(expert_probs)
        
        # 更新方差历史
        idx = self.history_idx % 100
        self.expert_variance_history[idx] = current_variance
        self.history_idx += 1
        
        # 计算平均方差
        if self.history_idx >= 100:
            avg_variance = self.expert_variance_history.mean()
            # 如果方差过大，增加负载均衡权重
            if avg_variance > 0.1:
                self.dynamic_balance_weight = torch.clamp(
                    self.dynamic_balance_weight * 1.01, max=5.0
                )
            else:
                self.dynamic_balance_weight = torch.clamp(
                    self.dynamic_balance_weight * 0.99, min=0.1
                )
    
    def get_expert_utilization(self):
        """获取专家利用率统计"""
        if self.total_tokens > 0:
            utilization = self.expert_counts / self.total_tokens
            return {
                'expert_counts': self.expert_counts.cpu().numpy(),
                'utilization': utilization.cpu().numpy(),
                'total_tokens': self.total_tokens.item(),
                'balance_score': 1.0 - utilization.std().item(),
                'dynamic_weight': self.dynamic_balance_weight.item(),
                'variance_history': self.expert_variance_history[:min(100, self.history_idx)].cpu().numpy()
            }
        return None


class ImprovedHomogeneousMoELayer(nn.Module):
    """改进的同构MoE层"""
    
    def __init__(self, d_model: int, num_experts: int = 4, d_ff: int = None,
                 top_k: int = 2, dropout: float = 0.1, load_balance_weight: float = 0.01):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # 创建专家
        self.experts = nn.ModuleList([
            ImprovedHomogeneousExpert(d_model, d_ff, dropout, expert_id=i)
            for i in range(num_experts)
        ])
        
        # 门控网络
        self.gate = AdaptiveGatingNetwork(
            d_model, num_experts, top_k, dropout
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 专家输出融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # 门控网络计算
        gates, indices, load_loss = self.gate(x, self.training)
        # gates: (batch_size, top_k), indices: (batch_size, top_k)
        
        # 初始化输出
        output = torch.zeros_like(x)  # (batch_size, seq_len, d_model)
        
        # 对每个样本处理
        for batch_idx in range(batch_size):
            sample_output = torch.zeros(seq_len, d_model, device=x.device)
            
            # 对该样本的top_k个专家进行处理
            for k in range(self.top_k):
                expert_idx = indices[batch_idx, k].item()
                expert_weight = gates[batch_idx, k].item()
                
                # 获取该样本的输入
                sample_input = x[batch_idx]  # (seq_len, d_model)
                
                # 专家处理
                expert_output = self.experts[expert_idx](sample_input)
                
                # 加权累加
                sample_output += expert_weight * expert_output
            
            output[batch_idx] = sample_output
        
        # 融合和归一化
        output = self.fusion_layer(output)
        output = self.norm(output + residual)
        
        return output, load_loss


class ImprovedHomogeneousMoEModel(BaseModel):
    """改进的同构MoE模型"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 3,
                 num_experts: int = 6, top_k: int = 2, num_classes: int = 2,
                 dropout: float = 0.1, max_length: int = 40, 
                 load_balance_weight: float = 0.01):
        super(ImprovedHomogeneousMoEModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        self.pos_encoding = self._create_pos_encoding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # MoE层
        self.moe_layers = nn.ModuleList([
            ImprovedHomogeneousMoELayer(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.final_norm = nn.LayerNorm(d_model)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
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
        batch_size, seq_len = x.shape
        
        # 嵌入和位置编码
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        
        # 通过MoE层
        total_load_loss = 0.0
        for moe_layer in self.moe_layers:
            x, load_loss = moe_layer(x)
            total_load_loss += load_loss
        
        # 最终归一化
        x = self.final_norm(x)
        
        # 全局平均池化（考虑padding）
        mask = (x.sum(dim=-1) != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(pooled)
        
        # 只返回logits，将load_loss存储为模型属性
        self.last_load_loss = total_load_loss * self.load_balance_weight
        return logits
    
    def get_expert_utilization(self):
        """获取所有层的专家利用率"""
        utilization_stats = {}
        
        for i, moe_layer in enumerate(self.moe_layers):
            stats = moe_layer.gate.get_expert_utilization()
            if stats:
                utilization_stats[f'layer_{i}'] = stats
        
        return utilization_stats
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ImprovedHomogeneousMoE',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'total_experts': self.num_layers * self.num_experts,
            'architecture': 'Improved Homogeneous (Enhanced Specialization + Adaptive Gating)'
        }