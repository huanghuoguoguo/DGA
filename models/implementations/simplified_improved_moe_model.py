#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 简化改进版MoE模型
基于实验分析的阶段1优化：简化门控网络，增强负载均衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class SimplifiedGateNetwork(nn.Module):
    """简化的门控网络"""
    
    def __init__(self, vocab_size, d_model=128, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # 简化的特征提取
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, 32, batch_first=True, bidirectional=True)
        
        # 简单的统计特征
        self.entropy_weight = nn.Parameter(torch.ones(1))
        self.length_weight = nn.Parameter(torch.ones(1))
        
        # 简化的门控网络 (64 -> 32 -> num_experts)
        self.gate_network = nn.Sequential(
            nn.Linear(64 + 2, 32),  # LSTM输出(64) + 统计特征(2)
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout防止过拟合
            nn.Linear(32, num_experts)
            # 不使用Softmax，在forward中使用Gumbel Softmax
        )
        
        # 温度参数，用于Gumbel Softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
    def compute_entropy(self, x):
        """计算序列熵（简化版）"""
        batch_size = x.size(0)
        entropies = []
        
        for i in range(batch_size):
            seq = x[i]
            valid_seq = seq[seq != 0]  # 去除padding
            
            if len(valid_seq) <= 1:
                entropies.append(0.0)
            else:
                # 计算字符频率
                unique_chars = len(torch.unique(valid_seq))
                max_entropy = math.log(min(unique_chars, 40))  # 最大可能熵
                
                # 简化的熵计算
                char_counts = torch.bincount(valid_seq, minlength=40).float()
                char_probs = char_counts / char_counts.sum()
                entropy = -torch.sum(char_probs * torch.log(char_probs + 1e-8))
                
                # 归一化熵
                normalized_entropy = entropy / (max_entropy + 1e-8)
                entropies.append(normalized_entropy.item())
        
        return torch.tensor(entropies, device=x.device).unsqueeze(1)
    
    def compute_length_feature(self, x):
        """计算长度特征（简化版）"""
        lengths = (x != 0).sum(dim=1).float()  # 有效长度
        normalized_lengths = lengths / 40.0  # 归一化到[0,1]
        return normalized_lengths.unsqueeze(1)
    
    def forward(self, x):
        # LSTM特征提取
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embedded)
        # 使用最后的隐藏状态
        lstm_features = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 64)
        
        # 简单统计特征
        entropy_features = self.compute_entropy(x)  # (batch, 1)
        length_features = self.compute_length_feature(x)  # (batch, 1)
        
        # 特征融合
        all_features = torch.cat([
            lstm_features,
            entropy_features * self.entropy_weight,
            length_features * self.length_weight
        ], dim=1)  # (batch, 66)
        
        # 门控权重计算
        gate_logits = self.gate_network(all_features)
        
        # 使用Gumbel Softmax增加随机性，防止过度集中
        if self.training:
            gate_weights = F.gumbel_softmax(gate_logits, tau=self.temperature, hard=False)
        else:
            gate_weights = F.softmax(gate_logits, dim=-1)
        
        return gate_weights


class SimplifiedExpert(nn.Module):
    """简化的专家网络基类"""
    
    def __init__(self, vocab_size, d_model=128, expert_type="base", dropout=0.1):
        super().__init__()
        self.expert_type = expert_type
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        raise NotImplementedError


class CNNExpert(SimplifiedExpert):
    """CNN专家 - 专注局部特征"""
    
    def __init__(self, vocab_size, d_model=128, dropout=0.1):
        super().__init__(vocab_size, d_model, "cnn", dropout)
        
        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, 64, kernel_size=k, padding=k//2)
            for k in [2, 3, 4]
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(64) for _ in range(3)
        ])
        
        self.output_proj = nn.Linear(64 * 3, d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(x)))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        features = torch.cat(conv_outputs, dim=1)
        return self.output_proj(features)


class LSTMExpert(SimplifiedExpert):
    """LSTM专家 - 专注序列依赖"""
    
    def __init__(self, vocab_size, d_model=128, dropout=0.1):
        super().__init__(vocab_size, d_model, "lstm", dropout)
        
        self.lstm = nn.LSTM(d_model, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.output_proj = nn.Linear(128, d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        return self.output_proj(weighted_output)


class MambaExpert(SimplifiedExpert):
    """Mamba专家 - 专注长序列建模"""
    
    def __init__(self, vocab_size, d_model=128, dropout=0.1):
        super().__init__(vocab_size, d_model, "mamba", dropout)
        
        # 简化的状态空间模块
        self.state_size = 32
        self.state_proj = nn.Linear(d_model, self.state_size)
        self.input_proj = nn.Linear(d_model, self.state_size)
        self.output_proj = nn.Linear(self.state_size, d_model)
        self.gate = nn.Linear(d_model, d_model)
        
        # 可学习的状态转移矩阵
        self.state_transition = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.01)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # 初始化状态
        h = torch.zeros(batch_size, self.state_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            # 状态更新
            h_proj = self.state_proj(x_t)
            x_proj = self.input_proj(x_t)
            h = torch.tanh(h @ self.state_transition.T + h_proj + x_proj)
            
            # 输出投影
            output_t = self.output_proj(h)
            
            # 门控机制
            gate_t = torch.sigmoid(self.gate(x_t))
            output_t = output_t * gate_t
            
            outputs.append(output_t)
        
        # 平均池化
        output = torch.stack(outputs, dim=1)
        output = torch.mean(output, dim=1)
        
        return self.norm(output)


class TransformerExpert(SimplifiedExpert):
    """Transformer专家 - 专注全局注意力"""
    
    def __init__(self, vocab_size, d_model=128, dropout=0.1):
        super().__init__(vocab_size, d_model, "transformer", dropout)
        
        # 简化的Transformer
        self.pos_encoding = self._create_pos_encoding(40, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 只用1层
        
        self.output_proj = nn.Linear(d_model, d_model)
        
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoding[:, :x.size(1), :].to(x.device)
        embedded = self.dropout(embedded)
        
        # 创建padding mask
        padding_mask = (x == 0)
        
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # 全局平均池化（忽略padding）
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return self.output_proj(pooled)


class SimplifiedImprovedMoEModel(BaseModel):
    """简化改进版MoE模型"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(SimplifiedImprovedMoEModel, self).__init__(vocab_size, num_classes)
        
        self.num_experts = 4
        self.d_model = d_model
        
        # 4个简化的专家
        self.experts = nn.ModuleList([
            CNNExpert(vocab_size, d_model, dropout),
            LSTMExpert(vocab_size, d_model, dropout),
            MambaExpert(vocab_size, d_model, dropout),
            TransformerExpert(vocab_size, d_model, dropout)
        ])
        
        # 简化的门控网络
        self.gate = SimplifiedGateNetwork(vocab_size, d_model, self.num_experts)
        
        # 最终分类器
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 损失权重
        self.load_balance_weight = 0.5  # 增加负载均衡权重
        self.diversity_weight = 0.2     # 添加多样性损失权重
        
    def compute_load_balance_loss(self, gate_weights):
        """计算负载均衡损失"""
        # 计算每个专家的平均使用率
        expert_usage = torch.mean(gate_weights, dim=0)  # (num_experts,)
        
        # 理想情况下每个专家使用率应该是 1/num_experts
        target_usage = 1.0 / self.num_experts
        
        # 使用均方误差作为负载均衡损失
        load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, target_usage))
        
        return load_balance_loss
    
    def compute_diversity_loss(self, expert_outputs):
        """计算专家输出多样性损失"""
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)
        
        # 计算专家输出之间的余弦相似度
        similarities = []
        for i in range(len(expert_outputs)):
            for j in range(i+1, len(expert_outputs)):
                # 计算余弦相似度
                sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=1).mean()
                similarities.append(sim)
        
        if similarities:
            # 多样性损失：相似度越高，损失越大
            diversity_loss = torch.stack(similarities).mean()
            return diversity_loss
        else:
            return torch.tensor(0.0, device=expert_outputs[0].device)
    
    def forward(self, x, return_gate_weights=False, return_expert_outputs=False):
        # 门控网络计算专家权重
        gate_weights = self.gate(x)  # (batch, num_experts)
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # (batch, d_model)
            expert_outputs.append(expert_output)
        
        # 加权混合
        mixed_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            mixed_output += gate_weights[:, i:i+1] * expert_output
        
        # 最终分类
        logits = self.classifier(mixed_output)
        
        # 返回不同的信息
        if return_expert_outputs:
            return logits, gate_weights, expert_outputs
        elif return_gate_weights:
            return logits, gate_weights
        else:
            return logits
    
    def get_gate_weights(self, x):
        """获取门控权重"""
        return self.gate(x)
    
    def get_expert_outputs(self, x):
        """获取各专家输出"""
        expert_outputs = {}
        for i, expert in enumerate(self.experts):
            expert_name = ['cnn', 'lstm', 'mamba', 'transformer'][i]
            expert_outputs[expert_name] = expert(x)
        return expert_outputs
    
    def compute_total_loss(self, logits, targets, gate_weights, expert_outputs):
        """计算总损失（分类损失 + 负载均衡损失 + 多样性损失）"""
        # 分类损失
        classification_loss = F.cross_entropy(logits, targets)
        
        # 负载均衡损失
        load_balance_loss = self.compute_load_balance_loss(gate_weights)
        
        # 多样性损失
        diversity_loss = self.compute_diversity_loss(expert_outputs)
        
        # 总损失
        total_loss = (classification_loss + 
                     self.load_balance_weight * load_balance_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'load_balance_loss': load_balance_loss.item(),
            'diversity_loss': diversity_loss.item()
        }