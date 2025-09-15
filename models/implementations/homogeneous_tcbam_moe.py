#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同构TCBAM-MoE模型实现
使用多个TCBAM专家的同构混合专家模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from models.implementations.tcbam_models import (
    ChannelAttention, SpatialAttention, CBAMBlock, 
    DPCNNBlock, BiLSTMAttention
)


class TCBAMExpert(nn.Module):
    """TCBAM专家模块"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, 
                 num_filters=128, num_heads=8, num_layers=2, dropout=0.1):
        super(TCBAMExpert, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CNN分支1 (kernel_size=2)
        self.conv1_k2 = nn.Conv1d(embed_dim, num_filters, kernel_size=2, padding=1)
        self.conv2_k2 = nn.Conv1d(num_filters, num_filters, kernel_size=2, padding=1)
        self.cbam1 = CBAMBlock(num_filters)
        self.dpcnn1 = DPCNNBlock(num_filters, num_filters, 2)
        self.bilstm_att1 = BiLSTMAttention(num_filters, hidden_dim//2, num_filters)
        
        # CNN分支2 (kernel_size=3)
        self.conv1_k3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv2_k3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3_k3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.cbam2 = CBAMBlock(num_filters)
        self.dpcnn2 = DPCNNBlock(num_filters, num_filters, 3)
        self.bilstm_att2 = BiLSTMAttention(num_filters, hidden_dim//2, num_filters)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_filters * 4, num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters * 2, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 嵌入 + 位置编码
        embedded = self.embedding(x)
        if seq_len <= self.pos_encoding.size(0):
            embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer编码
        padding_mask = (x == 0)
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # 转换为CNN输入格式
        cnn_input = transformer_out.transpose(1, 2)
        
        # 分支1: kernel_size=2
        c11 = F.relu(self.conv1_k2(cnn_input))
        c11 = self.dropout(c11)
        
        cbam1_out = self.cbam1(c11)
        c12 = F.relu(self.conv2_k2(c11))
        c12 = self.dropout(c12)
        
        # 维度匹配
        min_len = min(cbam1_out.size(2), c12.size(2))
        cbam1_out = cbam1_out[:, :, :min_len]
        c12 = c12[:, :, :min_len]
        residual1 = cbam1_out + c12
        dpcnn1_out = self.dpcnn1(residual1)
        
        bilstm1_input = c12.transpose(1, 2)
        bilstm1_out = self.bilstm_att1(bilstm1_input)
        
        # 分支2: kernel_size=3
        c21 = F.relu(self.conv1_k3(cnn_input))
        c21 = self.dropout(c21)
        
        c22 = F.relu(self.conv2_k3(c21))
        c22 = self.dropout(c22)
        
        c23 = F.relu(self.conv3_k3(c22))
        c23 = self.dropout(c23)
        
        cbam2_out = self.cbam2(c23)
        dpcnn2_out = self.dpcnn2(cbam2_out)
        
        bilstm2_input = c22.transpose(1, 2)
        bilstm2_out = self.bilstm_att2(bilstm2_input)
        
        # 特征融合
        dpcnn1_pooled = F.adaptive_avg_pool1d(dpcnn1_out, 1).squeeze(-1)
        dpcnn2_pooled = F.adaptive_avg_pool1d(dpcnn2_out, 1).squeeze(-1)
        
        combined_features = torch.cat([
            dpcnn1_pooled, bilstm1_out, bilstm2_out, dpcnn2_pooled
        ], dim=1)
        
        fused_features = self.feature_fusion(combined_features)
        return fused_features


class GatingNetwork(nn.Module):
    """门控网络"""
    
    def __init__(self, input_dim, num_experts, hidden_dim=256):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # 全局平均池化
        pooled = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        gate_logits = self.gate(pooled)  # (batch_size, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)
        return gate_weights


class HomogeneousTCBAMMoE(BaseModel):
    """同构TCBAM-MoE模型"""
    
    def __init__(self, vocab_size, num_classes=2, num_experts=4, 
                 embed_dim=128, hidden_dim=128, num_filters=128, 
                 num_heads=8, num_layers=2, dropout=0.1):
        super(HomogeneousTCBAMMoE, self).__init__(vocab_size, num_classes)
        
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        
        # 共享嵌入层
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # 门控网络
        self.gating_network = GatingNetwork(embed_dim, num_experts)
        
        # 多个TCBAM专家
        self.experts = nn.ModuleList([
            TCBAMExpert(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_filters=num_filters,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_classes)
        )
        
        # 负载均衡损失权重
        self.load_balance_weight = 0.01
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 共享嵌入
        embedded = self.shared_embedding(x)
        if seq_len <= self.pos_encoding.size(0):
            embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 门控网络计算专家权重
        gate_weights = self.gating_network(embedded)  # (batch_size, num_experts)
        
        # 所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch_size, num_filters)
            expert_outputs.append(expert_out)
        
        # 专家输出堆叠
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, num_filters)
        
        # 加权融合
        gate_weights = gate_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        fused_output = torch.sum(gate_weights * expert_outputs, dim=1)  # (batch_size, num_filters)
        
        # 分类
        logits = self.classifier(fused_output)
        
        # 存储门控权重用于分析
        self.last_gate_weights = gate_weights.squeeze(-1)
        
        return logits
    
    def get_load_balance_loss(self):
        """计算负载均衡损失"""
        if hasattr(self, 'last_gate_weights'):
            # 计算专家使用率的方差
            expert_usage = torch.mean(self.last_gate_weights, dim=0)  # (num_experts,)
            target_usage = 1.0 / self.num_experts
            load_balance_loss = torch.var(expert_usage) / (target_usage ** 2)
            return self.load_balance_weight * load_balance_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_expert_usage_stats(self):
        """获取专家使用统计"""
        if hasattr(self, 'last_gate_weights'):
            expert_usage = torch.mean(self.last_gate_weights, dim=0)
            return expert_usage.detach().cpu().numpy()
        return None
    
    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        print(f"📋 模型信息:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {model_size_mb:.2f} MB")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  类别数: {self.num_classes}")
        print(f"  专家数量: {self.num_experts}")
        print(f"  专家类型: TCBAM (同构)")