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
    """同构专家网络 - 使用BiLSTM+Attention架构"""
    
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int = None, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        if hidden_size is None:
            hidden_size = d_model // 2
        
        self.d_model = d_model
        self.hidden_size = hidden_size
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            d_model, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # 注意力机制
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_size * 2, d_model)
        
        # 专家特化参数
        self.expert_bias = nn.Parameter(torch.randn(d_model) * 0.01)
        self.expert_scale = nn.Parameter(torch.ones(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len) - 输入序列的token ids
        返回: (batch_size, d_model) - 专家输出
        """
        # 嵌入 (batch, seq_len, d_model)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # BiLSTM (batch, seq_len, hidden_size*2)
        lstm_out, _ = self.lstm(embedded)
        
        # 注意力权重 (batch, seq_len, 1)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # 加权求和 (batch, hidden_size*2)
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # 投影到d_model维度 (batch, d_model)
        output = self.output_proj(weighted_output)
        
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
    
    def __init__(self, vocab_size: int, d_model: int, num_experts: int = 4, 
                 top_k: int = 2, dropout: float = 0.1, load_balance_weight: float = 0.01):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.d_model = d_model
        
        # 创建同构专家 - 每个专家都是BiLSTM+Attention
        self.experts = nn.ModuleList([
            HomogeneousExpert(vocab_size, d_model, dropout=dropout)
            for _ in range(num_experts)
        ])
        
        # 门控网络 - 需要先将输入转换为特征表示
        self.input_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gate = GatingNetwork(d_model, num_experts, top_k, dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len) - token序列
        """
        batch_size, seq_len = x.shape
        
        # 将输入转换为特征表示用于门控网络
        embedded = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        # 全局平均池化得到序列级别的表示
        pooled_features = torch.mean(embedded, dim=1)  # (batch_size, d_model)
        
        # 门控选择
        gates, indices, load_loss = self.gate(pooled_features, self.training)
        
        # 批量处理所有专家
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch_size, d_model)
            expert_outputs.append(expert_out)
        
        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, d_model)
        
        # 选择top-k专家输出并加权求和
        output = torch.zeros(batch_size, self.d_model, device=x.device)
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # (batch_size,)
            gate_weight = gates[:, k]   # (batch_size,)
            
            # 选择对应专家的输出
            selected_output = expert_outputs[torch.arange(batch_size), expert_idx]  # (batch_size, d_model)
            output += gate_weight.unsqueeze(-1) * selected_output
        
        # 残差连接和层归一化
        output = self.norm(pooled_features + output)
        
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
        
        # MoE层
        self.moe_layers = nn.ModuleList([
            HomogeneousMoELayer(
                vocab_size=vocab_size,
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
        
        # 通过MoE层 - 每层直接处理token输入
        total_load_loss = 0.0
        current_input = x  # 保持token格式
        
        for i, moe_layer in enumerate(self.moe_layers):
            if i == 0:
                # 第一层直接处理token输入
                layer_output, load_loss = moe_layer(current_input)
            else:
                # 后续层需要将特征表示转换回token格式进行处理
                # 这里我们使用第一层的输出作为所有后续层的输入
                layer_output, load_loss = moe_layer(current_input)
            
            total_load_loss += load_loss
            # 保持输出格式为特征向量，但输入仍为token
        
        # 最终输出已经是特征向量格式 (batch_size, d_model)
        x = self.norm(layer_output)
        
        # 分类
        logits = self.classifier(x)
        
        # 只返回logits，负载均衡损失暂时忽略以保持与训练框架的兼容性
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
            'model_name': 'HomogeneousMoE',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'total_experts': self.num_layers * self.num_experts
        }