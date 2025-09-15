#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 专门化专家模型实现
针对字符级和字典级DGA的专门化专家，以及高级注意力模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class CBAMAttention(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels, reduction=16):
        super(CBAMAttention, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x


class CharacterLevelExpert(BaseModel):
    """字符级DGA专家模型 - 专门处理高熵随机字符序列"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(CharacterLevelExpert, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 多尺度卷积层 - 捕获不同长度的字符模式
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, 64, kernel_size=k, padding=k//2)
            for k in [2, 3, 4, 5, 6]  # 更多尺度，适合字符级模式
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(64) for _ in range(5)
        ])
        
        # CBAM注意力机制
        self.cbam = CBAMAttention(64 * 5)
        
        # 字符频率分析层
        self.char_freq_analyzer = nn.Linear(vocab_size, 32)
        
        # 熵计算模块
        self.entropy_weight = nn.Parameter(torch.ones(1))
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(64 * 5 + 32 + 1, d_model),  # conv + freq + entropy
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Linear(d_model // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def compute_char_frequency(self, x):
        """计算字符频率特征"""
        batch_size = x.size(0)
        char_freq = torch.zeros(batch_size, self.vocab_size, device=x.device)
        
        for i in range(batch_size):
            seq = x[i]
            valid_seq = seq[seq != 0]  # 去除padding
            if len(valid_seq) > 0:
                for char_idx in valid_seq:
                    char_freq[i, char_idx] += 1
                char_freq[i] = char_freq[i] / len(valid_seq)  # 归一化
        
        return self.char_freq_analyzer(char_freq)
    
    def compute_entropy(self, x):
        """计算序列熵"""
        batch_size = x.size(0)
        entropies = []
        
        for i in range(batch_size):
            seq = x[i]
            valid_seq = seq[seq != 0]
            
            if len(valid_seq) <= 1:
                entropies.append(0.0)
            else:
                char_counts = torch.bincount(valid_seq, minlength=self.vocab_size).float()
                char_probs = char_counts / char_counts.sum()
                entropy = -torch.sum(char_probs * torch.log(char_probs + 1e-8))
                entropies.append(entropy.item())
        
        return torch.tensor(entropies, device=x.device).unsqueeze(1)
    
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, d_model)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # 多尺度卷积特征提取
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(embedded)))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        conv_features = torch.cat(conv_outputs, dim=1)  # (batch, 64*5)
        
        # CBAM注意力
        conv_features_expanded = conv_features.unsqueeze(2)  # (batch, 64*5, 1)
        attended_features = self.cbam(conv_features_expanded).squeeze(2)
        
        # 字符频率特征
        freq_features = self.compute_char_frequency(x)  # (batch, 32)
        
        # 熵特征
        entropy_features = self.compute_entropy(x) * self.entropy_weight  # (batch, 1)
        
        # 特征融合
        all_features = torch.cat([attended_features, freq_features, entropy_features], dim=1)
        fused_features = self.feature_fusion(all_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits


class DictionaryLevelExpert(BaseModel):
    """字典级DGA专家模型 - 专门处理基于词典的序列"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(DictionaryLevelExpert, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # BiGRU层 - 捕获双向序列依赖
        self.bigru = nn.GRU(d_model, 64, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        
        # 多头自注意力机制
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=128,  # BiGRU输出维度
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(40, 128)
        
        # 词汇相似度分析
        self.word_similarity = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 语义分析层
        self.semantic_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # 长度分析
        self.length_analyzer = nn.Linear(1, 16)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 + 32 + 32 + 16, d_model),  # attention + similarity + semantic + length
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Linear(d_model // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def compute_length_features(self, x):
        """计算长度特征"""
        lengths = (x != 0).sum(dim=1).float().unsqueeze(1)  # (batch, 1)
        return self.length_analyzer(lengths / 40.0)  # 归一化
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, d_model)
        embedded = self.dropout(embedded)
        
        # BiGRU
        gru_out, _ = self.bigru(embedded)  # (batch, seq_len, 128)
        
        # 位置编码
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        gru_out = gru_out + pos_enc
        
        # 多头自注意力
        attn_out, attn_weights = self.multihead_attention(gru_out, gru_out, gru_out)
        
        # 创建padding mask
        padding_mask = (x == 0)
        
        # 加权平均池化（忽略padding）
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled_attn = (attn_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 词汇相似度特征
        similarity_features = self.word_similarity(pooled_attn)
        
        # 语义特征
        semantic_features = self.semantic_analyzer(pooled_attn)
        
        # 长度特征
        length_features = self.compute_length_features(x)
        
        # 特征融合
        all_features = torch.cat([pooled_attn, similarity_features, semantic_features, length_features], dim=1)
        fused_features = self.feature_fusion(all_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits


class BiGRUAttentionExpert(BaseModel):
    """BiGRU + Attention专家模型"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(BiGRUAttentionExpert, self).__init__(vocab_size, num_classes)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # BiGRU层
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # BiGRU
        gru_out, _ = self.bigru(embedded)  # (batch, seq_len, 128)
        
        # 注意力权重计算
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        weighted_output = torch.sum(gru_out * attention_weights, dim=1)  # (batch, 128)
        
        # 分类
        logits = self.classifier(weighted_output)
        
        return logits


class CNNWithCBAMExpert(BaseModel):
    """CNN + CBAM注意力专家模型"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super(CNNWithCBAMExpert, self).__init__(vocab_size, num_classes)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, 128, kernel_size=k, padding=k//2)
            for k in [2, 3, 4, 5]
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(128) for _ in range(4)
        ])
        
        # CBAM注意力
        self.cbam = CBAMAttention(128 * 4)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # 多尺度卷积
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = F.relu(bn(conv(embedded)))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        features = torch.cat(conv_outputs, dim=1)  # (batch, 128*4)
        
        # CBAM注意力
        features_expanded = features.unsqueeze(2)  # (batch, 128*4, 1)
        attended_features = self.cbam(features_expanded).squeeze(2)
        
        # 分类
        logits = self.classifier(attended_features)
        
        return logits


class TransformerExpert(BaseModel):
    """Transformer专家模型"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1, num_heads=8, num_layers=2):
        super(TransformerExpert, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_positional_encoding(40, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):
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
        
        # Transformer编码
        transformer_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # 全局平均池化（忽略padding）
        mask = (~padding_mask).float().unsqueeze(-1)
        pooled = (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits