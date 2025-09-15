#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版同构TCBAM-MoE模型实现
改进门控网络和负载均衡机制
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


class EnhancedTCBAMExpert(nn.Module):
    """增强版TCBAM专家模块"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, 
                 num_filters=128, num_heads=8, num_layers=2, dropout=0.1, expert_id=0):
        super(EnhancedTCBAMExpert, self).__init__()
        
        self.expert_id = expert_id
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        
        # 专家特异性初始化 - 每个专家有不同的初始化策略
        self.expert_bias = nn.Parameter(torch.randn(1) * 0.1)
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Transformer编码器 - 每个专家有不同的层数
        actual_layers = max(1, num_layers + (expert_id - 2))  # 专家0:1层, 专家1:2层, 专家2:3层, 专家3:4层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=actual_layers)
        
        # CNN分支1 (kernel_size=2) - 专家0和1更关注短模式
        if expert_id < 2:
            self.conv1_k2 = nn.Conv1d(embed_dim, num_filters, kernel_size=2, padding=1)
            self.conv2_k2 = nn.Conv1d(num_filters, num_filters, kernel_size=2, padding=1)
        else:
            self.conv1_k2 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
            self.conv2_k2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
            
        self.cbam1 = CBAMBlock(num_filters)
        self.dpcnn1 = DPCNNBlock(num_filters, num_filters, 2 if expert_id < 2 else 3)
        self.bilstm_att1 = BiLSTMAttention(num_filters, hidden_dim//2, num_filters)
        
        # CNN分支2 (kernel_size=3) - 专家2和3更关注长模式
        if expert_id >= 2:
            self.conv1_k3 = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)
            self.conv2_k3 = nn.Conv1d(num_filters, num_filters, kernel_size=4, padding=2)
            self.conv3_k3 = nn.Conv1d(num_filters, num_filters, kernel_size=4, padding=2)
        else:
            self.conv1_k3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
            self.conv2_k3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
            self.conv3_k3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
            
        self.cbam2 = CBAMBlock(num_filters)
        self.dpcnn2 = DPCNNBlock(num_filters, num_filters, 3 if expert_id < 2 else 4)
        self.bilstm_att2 = BiLSTMAttention(num_filters, hidden_dim//2, num_filters)
        
        # 特征融合 - 每个专家有不同的融合策略
        fusion_dim = num_filters * 4
        if expert_id % 2 == 0:  # 专家0,2使用更深的融合网络
            self.feature_fusion = nn.Sequential(
                nn.Linear(fusion_dim, num_filters * 3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_filters * 3, num_filters * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_filters * 2, num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:  # 专家1,3使用较浅的融合网络
            self.feature_fusion = nn.Sequential(
                nn.Linear(fusion_dim, num_filters * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_filters * 2, num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # 专家特异性参数初始化
        self._init_expert_weights()
        
    def _init_expert_weights(self):
        """专家特异性权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # 每个专家使用不同的初始化策略
                if self.expert_id == 0:
                    nn.init.xavier_uniform_(param)
                elif self.expert_id == 1:
                    nn.init.kaiming_uniform_(param)
                elif self.expert_id == 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.kaiming_normal_(param)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 嵌入 + 位置编码
        embedded = self.embedding(x)
        if seq_len <= self.pos_encoding.size(0):
            embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 添加专家偏置
        embedded = embedded + self.expert_bias
        
        # Transformer编码
        transformer_out = self.transformer_encoder(embedded)
        
        # 转换为CNN输入格式
        cnn_input = transformer_out.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        # CNN分支1
        conv1_out = F.relu(self.conv1_k2(cnn_input))
        conv1_out = F.relu(self.conv2_k2(conv1_out))
        conv1_out = self.cbam1(conv1_out)
        conv1_out = self.dpcnn1(conv1_out)
        
        # 转换为BiLSTM输入格式
        conv1_lstm_input = conv1_out.transpose(1, 2)
        conv1_final = self.bilstm_att1(conv1_lstm_input)
        
        # CNN分支2
        conv2_out = F.relu(self.conv1_k3(cnn_input))
        conv2_out = F.relu(self.conv2_k3(conv2_out))
        conv2_out = F.relu(self.conv3_k3(conv2_out))
        conv2_out = self.cbam2(conv2_out)
        conv2_out = self.dpcnn2(conv2_out)
        
        # 转换为BiLSTM输入格式
        conv2_lstm_input = conv2_out.transpose(1, 2)
        conv2_final = self.bilstm_att2(conv2_lstm_input)
        
        # 全局池化
        transformer_pooled = torch.mean(transformer_out, dim=1)
        
        # 特征拼接
        combined_features = torch.cat([
            transformer_pooled,
            conv1_final,
            conv2_final,
            transformer_pooled * 0.5  # 添加额外的全局特征
        ], dim=1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features


class EnhancedGatingNetwork(nn.Module):
    """增强版门控网络"""
    
    def __init__(self, input_dim, num_experts, hidden_dim=256):
        super(EnhancedGatingNetwork, self).__init__()
        self.num_experts = num_experts
        
        # 多层感知机门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_experts)
        )
        
        # 特征分析网络 - 分析输入特征的类型
        self.feature_analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4种特征类型：字符级、字典级、结构级、混合级
        )
        
        # 专家-特征类型映射权重
        self.expert_feature_weights = nn.Parameter(torch.randn(num_experts, 4))
        
        # 温度参数用于控制门控的锐度
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size = x.size(0)
        
        # 多种池化策略
        mean_pooled = torch.mean(x, dim=1)  # 全局平均池化
        max_pooled, _ = torch.max(x, dim=1)  # 全局最大池化
        
        # 注意力池化
        attention_weights = F.softmax(torch.sum(x, dim=-1), dim=-1)  # (batch_size, seq_len)
        attention_pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        # 组合特征
        combined_features = torch.cat([mean_pooled, max_pooled, attention_pooled], dim=-1)
        
        # 基础门控权重
        gate_logits = self.gate(mean_pooled)  # (batch_size, num_experts)
        
        # 特征类型分析
        feature_type_logits = self.feature_analyzer(mean_pooled)  # (batch_size, 4)
        feature_type_probs = F.softmax(feature_type_logits, dim=-1)
        
        # 专家-特征类型匹配分数
        expert_feature_scores = torch.matmul(feature_type_probs, self.expert_feature_weights.t())  # (batch_size, num_experts)
        
        # 组合门控分数
        final_gate_logits = gate_logits + expert_feature_scores
        
        # 使用温度参数调节锐度
        final_gate_logits = final_gate_logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # 计算最终权重
        gate_weights = F.softmax(final_gate_logits, dim=-1)
        
        return gate_weights, feature_type_probs


class EnhancedHomogeneousTCBAMMoE(BaseModel):
    """增强版同构TCBAM-MoE模型"""
    
    def __init__(self, vocab_size, num_classes=2, num_experts=4, 
                 embed_dim=128, hidden_dim=128, num_filters=128, 
                 num_heads=8, num_layers=2, dropout=0.1):
        super(EnhancedHomogeneousTCBAMMoE, self).__init__(vocab_size, num_classes)
        
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        
        # 共享嵌入层
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # 增强门控网络
        self.gating_network = EnhancedGatingNetwork(embed_dim, num_experts)
        
        # 多个增强TCBAM专家
        self.experts = nn.ModuleList([
            EnhancedTCBAMExpert(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_filters=num_filters,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                expert_id=i
            ) for i in range(num_experts)
        ])
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.LayerNorm(num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_filters // 4),
            nn.ReLU(),
            nn.Linear(num_filters // 4, num_classes)
        )
        
        # 负载均衡损失权重
        self.load_balance_weight = 0.01
        self.diversity_weight = 0.01
        
        # 专家使用统计
        self.expert_usage_count = torch.zeros(num_experts)
        self.total_samples = 0
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 共享嵌入
        embedded = self.shared_embedding(x)
        seq_len = embedded.size(1)
        if seq_len <= self.pos_encoding.size(0):
            embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 门控网络
        gate_weights, feature_type_probs = self.gating_network(embedded)
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # 堆叠专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, num_filters)
        
        # 加权组合
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        combined_output = torch.sum(expert_outputs * gate_weights_expanded, dim=1)  # (batch_size, num_filters)
        
        # 分类
        logits = self.classifier(combined_output)
        
        # 更新专家使用统计
        if self.training:
            with torch.no_grad():
                expert_usage = torch.sum(gate_weights, dim=0)
                self.expert_usage_count += expert_usage.cpu()
                self.total_samples += batch_size
        
        # 存储中间结果用于损失计算
        self.last_gate_weights = gate_weights
        self.last_expert_outputs = expert_outputs
        self.last_feature_type_probs = feature_type_probs
        
        return logits
    
    def get_load_balance_loss(self):
        """计算负载均衡损失"""
        if not hasattr(self, 'last_gate_weights'):
            return torch.tensor(0.0)
        
        # 计算专家使用的方差
        expert_usage = torch.mean(self.last_gate_weights, dim=0)
        target_usage = 1.0 / self.num_experts
        balance_loss = torch.var(expert_usage) + torch.mean((expert_usage - target_usage) ** 2)
        
        return balance_loss * self.load_balance_weight
    
    def get_diversity_loss(self):
        """计算专家多样性损失"""
        if not hasattr(self, 'last_expert_outputs'):
            return torch.tensor(0.0)
        
        # 计算专家输出的相似性
        expert_outputs = self.last_expert_outputs  # (batch_size, num_experts, num_filters)
        
        # 计算专家间的余弦相似性
        normalized_outputs = F.normalize(expert_outputs, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized_outputs, normalized_outputs.transpose(-1, -2))
        
        # 去除对角线元素（自相似性）
        mask = torch.eye(self.num_experts, device=similarity_matrix.device).unsqueeze(0)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # 多样性损失：鼓励专家输出不同
        diversity_loss = torch.mean(torch.abs(similarity_matrix))
        
        return diversity_loss * self.diversity_weight
    
    def get_expert_usage_stats(self):
        """获取专家使用统计"""
        if self.total_samples == 0:
            return None
        
        usage_percentages = (self.expert_usage_count / self.total_samples * 100).tolist()
        return {
            'expert_usage_percentages': usage_percentages,
            'total_samples': self.total_samples,
            'usage_variance': float(torch.var(self.expert_usage_count / self.total_samples))
        }
    
    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"📋 模型信息:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  类别数: {self.num_classes}")
        print(f"  专家数量: {self.num_experts}")
        print(f"  专家类型: Enhanced TCBAM (同构增强)")
        print(f"  负载均衡权重: {self.load_balance_weight}")
        print(f"  多样性损失权重: {self.diversity_weight}")
    
    def get_model_info(self):
        """获取模型信息字典"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'EnhancedHomogeneousTCBAMMoE',
            'total_params': total_params,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'num_experts': self.num_experts,
            'embed_dim': self.embed_dim
        }