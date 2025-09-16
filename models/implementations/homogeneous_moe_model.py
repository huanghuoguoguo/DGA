#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同构混合专家模型(Homogeneous MoE)
使用相同架构的专家网络，通过门控网络进行选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class HomogeneousExpert(nn.Module):
    """同构专家网络 - 使用相同的架构"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.expert_net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 专家特化参数
        self.expert_bias = nn.Parameter(torch.randn(d_model) * 0.01)
        self.expert_scale = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        # 基础变换
        output = self.expert_net(x)
        
        # 专家特化
        output = output * self.expert_scale + self.expert_bias
        
        return output


class GatingNetwork(nn.Module):
    """门控网络 - 决定专家选择"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, 
                 dropout: float = 0.1, noise_std: float = 0.1):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_experts)
        )
        
        # 负载均衡相关
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
    def forward(self, x, training=True):
        """
        x: (batch_size * seq_len, d_model) - 已经是扁平化的输入
        返回: (gates, indices, load_balancing_loss)
        """
        # x已经是扁平化的，直接使用
        x_flat = x  # (batch_size * seq_len, d_model)
        
        # 计算门控分数
        gate_logits = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        
        # 添加噪声（训练时）
        if training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k选择
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # 计算门控权重
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # 负载均衡损失
        load_balancing_loss = self._compute_load_balancing_loss(gate_logits)
        
        # 更新专家使用统计
        if training:
            self._update_expert_stats(top_k_indices)
        
        return top_k_gates, top_k_indices, load_balancing_loss
    
    def _compute_load_balancing_loss(self, gate_logits):
        """计算负载均衡损失"""
        # 计算每个专家的平均概率
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_probs = gate_probs.mean(dim=0)  # (num_experts,)
        
        # 理想情况下每个专家的概率应该是 1/num_experts
        target_prob = 1.0 / self.num_experts
        
        # 计算KL散度作为负载均衡损失
        load_loss = F.kl_div(
            torch.log(expert_probs + 1e-8),
            torch.full_like(expert_probs, target_prob),
            reduction='sum'
        )
        
        return load_loss
    
    def _update_expert_stats(self, top_k_indices):
        """更新专家使用统计"""
        # 统计每个专家被选择的次数
        for expert_idx in range(self.num_experts):
            count = (top_k_indices == expert_idx).sum().float()
            self.expert_counts[expert_idx] += count
        
        self.total_tokens += top_k_indices.numel()
    
    def get_expert_utilization(self):
        """获取专家利用率统计"""
        if self.total_tokens > 0:
            utilization = self.expert_counts / self.total_tokens
            return {
                'expert_counts': self.expert_counts.cpu().numpy(),
                'utilization': utilization.cpu().numpy(),
                'total_tokens': self.total_tokens.item(),
                'balance_score': 1.0 - utilization.std().item()  # 越接近1越均衡
            }
        return None


class HomogeneousMoELayer(nn.Module):
    """同构MoE层"""
    
    def __init__(self, d_model: int, num_experts: int = 4, d_ff: int = None,
                 top_k: int = 2, dropout: float = 0.1, load_balance_weight: float = 0.01):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        # 创建同构专家
        self.experts = nn.ModuleList([
            HomogeneousExpert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = GatingNetwork(d_model, num_experts, top_k, dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # 门控选择
        gates, indices, load_loss = self.gate(x_flat, self.training)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 批量处理所有专家
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x_flat)  # (batch_size * seq_len, d_model)
            expert_outputs.append(expert_out)
        
        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size * seq_len, num_experts, d_model)
        
        # 选择top-k专家输出并加权求和
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # (batch_size * seq_len,)
            gate_weight = gates[:, k]   # (batch_size * seq_len,)
            
            # 选择对应专家的输出
            selected_output = expert_outputs[torch.arange(x_flat.size(0)), expert_idx]  # (batch_size * seq_len, d_model)
            output += gate_weight.unsqueeze(-1) * selected_output
        
        # 残差连接和层归一化
        output = output.view(batch_size, seq_len, d_model)
        output = self.norm(x + output)
        
        return output, load_loss


class HomogeneousMoEModel(nn.Module):
    """同构MoE模型用于DGA检测"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 3,
                 num_experts: int = 4, top_k: int = 2, num_classes: int = 2,
                 dropout: float = 0.1, max_length: int = 40, 
                 load_balance_weight: float = 0.01):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # MoE层
        self.moe_layers = nn.ModuleList([
            HomogeneousMoELayer(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight
            )
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(d_model)
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
        batch_size, seq_len = x.shape
        
        # 嵌入和位置编码
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        else:
            # 如果序列太长，截断位置编码
            x = x + self.pos_encoding.unsqueeze(0)
        
        x = self.dropout(x)
        
        # 通过MoE层
        total_load_loss = 0.0
        for moe_layer in self.moe_layers:
            x, load_loss = moe_layer(x)
            total_load_loss += load_loss
        
        # 全局平均池化
        x = self.norm(x)
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(x)
        
        # 始终返回logits和负载均衡损失以保持一致性
        return logits, total_load_loss * self.load_balance_weight
    
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
            'model_name': 'HomogeneousMoE',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'total_experts': self.num_layers * self.num_experts
        }