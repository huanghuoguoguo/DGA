#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 改进版MoE模型实现
基于分析报告的第一阶段优化：扩展专家数量，改进门控网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from .cnn_model import CNNModel
from .lstm_model import LSTMModel
from .mamba_model import MambaModel


class EntropyCalculator(nn.Module):
    """熵计算模块"""
    
    def __init__(self, vocab_size=40):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, x):
        """计算序列的字符熵"""
        batch_size = x.size(0)
        entropy_features = []
        
        for i in range(batch_size):
            # 计算字符频率
            seq = x[i]
            char_counts = torch.bincount(seq, minlength=self.vocab_size).float()
            char_probs = char_counts / (char_counts.sum() + 1e-8)
            
            # 计算熵
            entropy = -torch.sum(char_probs * torch.log(char_probs + 1e-8))
            entropy_features.append(entropy)
        
        return torch.stack(entropy_features).unsqueeze(1)  # (batch, 1)


class LengthAnalyzer(nn.Module):
    """长度分析模块"""
    
    def __init__(self, max_length=40):
        super().__init__()
        self.max_length = max_length
        
    def forward(self, x):
        """分析序列长度特征"""
        # 计算有效长度（非padding）
        lengths = (x != 0).sum(dim=1).float()  # (batch,)
        
        # 归一化长度
        normalized_lengths = lengths / self.max_length
        
        # 长度分类特征
        short_mask = (lengths <= 8).float()
        medium_mask = ((lengths > 8) & (lengths <= 16)).float()
        long_mask = (lengths > 16).float()
        
        length_features = torch.stack([
            normalized_lengths,
            short_mask,
            medium_mask,
            long_mask
        ], dim=1)  # (batch, 4)
        
        return length_features


class CharDistributionAnalyzer(nn.Module):
    """字符分布分析模块"""
    
    def __init__(self, vocab_size=40):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, x):
        """分析字符分布特征"""
        batch_size = x.size(0)
        dist_features = []
        
        for i in range(batch_size):
            seq = x[i]
            # 过滤padding
            valid_seq = seq[seq != 0]
            
            if len(valid_seq) == 0:
                # 空序列处理
                features = torch.zeros(6)
            else:
                # 字符频率分布
                char_counts = torch.bincount(valid_seq, minlength=self.vocab_size).float()
                char_probs = char_counts / char_counts.sum()
                
                # 统计特征
                unique_chars = (char_counts > 0).sum().float()
                max_freq = char_probs.max()
                min_freq = char_probs[char_probs > 0].min() if (char_probs > 0).sum() > 0 else 0
                
                # 字符类型统计（假设1-26是字母，27-36是数字）
                letter_ratio = char_probs[1:27].sum()  # 字母比例
                digit_ratio = char_probs[27:37].sum()   # 数字比例
                special_ratio = char_probs[37:].sum()   # 特殊字符比例
                
                features = torch.tensor([
                    unique_chars / self.vocab_size,  # 唯一字符比例
                    max_freq,                         # 最大频率
                    min_freq,                         # 最小频率
                    letter_ratio,                     # 字母比例
                    digit_ratio,                      # 数字比例
                    special_ratio                     # 特殊字符比例
                ])
            
            dist_features.append(features)
        
        return torch.stack(dist_features)  # (batch, 6)


class PatternDetector(nn.Module):
    """模式检测模块"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """检测序列模式特征"""
        batch_size = x.size(0)
        pattern_features = []
        
        for i in range(batch_size):
            seq = x[i]
            valid_seq = seq[seq != 0]
            
            if len(valid_seq) <= 1:
                features = torch.zeros(4)
            else:
                # 重复字符检测
                repeat_count = 0
                for j in range(len(valid_seq) - 1):
                    if valid_seq[j] == valid_seq[j + 1]:
                        repeat_count += 1
                repeat_ratio = repeat_count / (len(valid_seq) - 1)
                
                # 递增/递减模式检测
                inc_count = 0
                dec_count = 0
                for j in range(len(valid_seq) - 1):
                    if valid_seq[j + 1] > valid_seq[j]:
                        inc_count += 1
                    elif valid_seq[j + 1] < valid_seq[j]:
                        dec_count += 1
                
                inc_ratio = inc_count / (len(valid_seq) - 1)
                dec_ratio = dec_count / (len(valid_seq) - 1)
                
                # 周期性检测（简单版本）
                periodicity = 0
                if len(valid_seq) >= 4:
                    # 检测长度为2的周期
                    period2_matches = 0
                    for j in range(len(valid_seq) - 2):
                        if valid_seq[j] == valid_seq[j + 2]:
                            period2_matches += 1
                    periodicity = period2_matches / max(1, len(valid_seq) - 2)
                
                features = torch.tensor([
                    repeat_ratio,   # 重复字符比例
                    inc_ratio,      # 递增比例
                    dec_ratio,      # 递减比例
                    periodicity     # 周期性
                ])
            
            pattern_features.append(features)
        
        return torch.stack(pattern_features)  # (batch, 4)


class ImprovedGateNetwork(nn.Module):
    """改进的门控网络"""
    
    def __init__(self, vocab_size, d_model=128, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # 基础嵌入
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 特征提取器
        self.entropy_calculator = EntropyCalculator(vocab_size)
        self.length_analyzer = LengthAnalyzer()
        self.char_dist_analyzer = CharDistributionAnalyzer(vocab_size)
        self.pattern_detector = PatternDetector()
        
        # 序列特征提取
        self.lstm = nn.LSTM(d_model, 64, batch_first=True, bidirectional=True)
        
        # 特征融合
        feature_dim = 128 + 1 + 4 + 6 + 4  # lstm(128) + entropy(1) + length(4) + char_dist(6) + pattern(4)
        
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # 基础序列特征
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embedded)
        seq_features = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 128)
        
        # 统计特征
        entropy_features = self.entropy_calculator(x)      # (batch, 1)
        length_features = self.length_analyzer(x)          # (batch, 4)
        char_dist_features = self.char_dist_analyzer(x)    # (batch, 6)
        pattern_features = self.pattern_detector(x)        # (batch, 4)
        
        # 特征融合
        all_features = torch.cat([
            seq_features,
            entropy_features,
            length_features,
            char_dist_features,
            pattern_features
        ], dim=1)  # (batch, 143)
        
        # 门控权重计算
        gate_weights = self.gate_network(all_features)
        
        return gate_weights


class TransformerExpert(nn.Module):
    """Transformer专家（简化版）"""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_pos_encoding(40, d_model)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        # 嵌入和位置编码
        embedded = self.embedding(x)
        embedded = embedded * math.sqrt(embedded.size(-1))
        embedded = embedded + self.pos_encoding[:, :x.size(1), :].to(x.device)
        embedded = self.dropout(embedded)
        
        # Transformer编码
        # 创建padding mask
        padding_mask = (x == 0)
        
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # 全局平均池化（忽略padding）
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1)
        
        return self.output_proj(pooled)


class ImprovedMoEModel(BaseModel):
    """改进版混合专家模型（4个专家）"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(ImprovedMoEModel, self).__init__(vocab_size, num_classes)
        
        self.num_experts = 4
        self.d_model = d_model
        
        # 4个专家网络
        self.cnn_expert = self._create_cnn_expert(vocab_size, d_model, dropout)
        self.lstm_expert = self._create_lstm_expert(vocab_size, d_model, dropout)
        self.mamba_expert = self._create_mamba_expert(vocab_size, d_model, dropout)
        self.transformer_expert = TransformerExpert(vocab_size, d_model, dropout=dropout)
        
        # 改进的门控网络
        self.gate = ImprovedGateNetwork(vocab_size, d_model, self.num_experts)
        
        # 最终分类器
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 负载均衡权重
        self.load_balance_weight = 0.1
        
    def _create_cnn_expert(self, vocab_size, d_model, dropout):
        """创建CNN专家"""
        class CNNExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.conv_layers = nn.ModuleList([
                    nn.Conv1d(d_model, 64, kernel_size=k, padding=k//2)
                    for k in [2, 3, 4, 5]  # 更多尺度
                ])
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(64) for _ in range(4)
                ])
                self.dropout = nn.Dropout(dropout)
                self.output_proj = nn.Linear(64 * 4, d_model)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.dropout(x)
                x = x.transpose(1, 2)
                
                conv_outputs = []
                for conv, bn in zip(self.conv_layers, self.batch_norms):
                    conv_out = F.relu(bn(conv(x)))
                    pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
                    conv_outputs.append(pooled.squeeze(2))
                
                features = torch.cat(conv_outputs, dim=1)
                return self.output_proj(features)
        
        return CNNExpert()
    
    def _create_lstm_expert(self, vocab_size, d_model, dropout):
        """创建LSTM专家"""
        class LSTMExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.lstm = nn.LSTM(d_model, 64, batch_first=True, bidirectional=True)
                self.attention = nn.Linear(128, 1)
                self.dropout = nn.Dropout(dropout)
                self.output_proj = nn.Linear(128, d_model)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.dropout(x)
                
                lstm_out, _ = self.lstm(x)
                attention_weights = F.softmax(self.attention(lstm_out), dim=1)
                weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
                
                return self.output_proj(weighted_output)
        
        return LSTMExpert()
    
    def _create_mamba_expert(self, vocab_size, d_model, dropout):
        """创建Mamba专家（简化版）"""
        class MambaExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                
                # 简化的状态空间模块
                self.state_proj = nn.Linear(d_model, 32)
                self.input_proj = nn.Linear(d_model, 32)
                self.output_proj = nn.Linear(32, d_model)
                self.gate = nn.Linear(d_model, d_model)
                
                self.norm = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)
                
                # 状态转移矩阵
                self.state_transition = nn.Parameter(torch.randn(32, 32) * 0.01)
                
            def forward(self, x):
                batch_size, seq_len = x.shape
                
                # 嵌入
                embedded = self.embedding(x)
                embedded = self.dropout(embedded)
                
                # 初始化状态
                h = torch.zeros(batch_size, 32, device=x.device)
                
                outputs = []
                for t in range(seq_len):
                    x_t = embedded[:, t, :]
                    
                    # 状态更新
                    h_proj = self.state_proj(x_t)
                    x_proj = self.input_proj(x_t)
                    h = torch.tanh(h @ self.state_transition.T + h_proj + x_proj)
                    
                    # 输出投影
                    output_t = self.output_proj(h)
                    
                    # 门控
                    gate_t = torch.sigmoid(self.gate(x_t))
                    output_t = output_t * gate_t
                    
                    outputs.append(output_t)
                
                # 平均池化
                output = torch.stack(outputs, dim=1)
                output = torch.mean(output, dim=1)
                
                return self.norm(output)
        
        return MambaExpert()
    
    def compute_load_balance_loss(self, gate_weights):
        """计算负载均衡损失"""
        # 计算每个专家的平均使用率
        expert_usage = torch.mean(gate_weights, dim=0)  # (num_experts,)
        
        # 理想情况下每个专家使用率应该是 1/num_experts
        target_usage = 1.0 / self.num_experts
        
        # 计算使用率方差作为负载均衡损失
        load_balance_loss = torch.var(expert_usage)
        
        return load_balance_loss
    
    def forward(self, x, return_gate_weights=False):
        # 门控网络计算专家权重
        gate_weights = self.gate(x)  # (batch, num_experts)
        
        # 专家输出
        cnn_output = self.cnn_expert(x)           # (batch, d_model)
        lstm_output = self.lstm_expert(x)         # (batch, d_model)
        mamba_output = self.mamba_expert(x)       # (batch, d_model)
        transformer_output = self.transformer_expert(x)  # (batch, d_model)
        
        # 加权混合
        mixed_output = (gate_weights[:, 0:1] * cnn_output + 
                       gate_weights[:, 1:2] * lstm_output +
                       gate_weights[:, 2:3] * mamba_output +
                       gate_weights[:, 3:4] * transformer_output)
        
        # 最终分类
        logits = self.classifier(mixed_output)
        
        if return_gate_weights:
            return logits, gate_weights
        return logits
    
    def get_gate_weights(self, x):
        """获取门控权重（用于分析）"""
        return self.gate(x)
    
    def get_expert_outputs(self, x):
        """获取各专家输出（用于分析）"""
        return {
            'cnn': self.cnn_expert(x),
            'lstm': self.lstm_expert(x),
            'mamba': self.mamba_expert(x),
            'transformer': self.transformer_expert(x)
        }