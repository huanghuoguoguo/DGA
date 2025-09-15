#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCBAM模型实现 - 改造版
Transformer + CNN + BiLSTM + Attention + CBAM的组合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    """CBAM注意力块"""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DPCNNBlock(nn.Module):
    """深度金字塔CNN块"""
    
    def __init__(self, in_channels, num_filters, kernel_size):
        super(DPCNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 限制最大迭代次数，避免无限循环
        max_iterations = 5
        iteration = 0
        
        while x.size(2) > 2 and iteration < max_iterations:
            # 下采样层
            p1 = self.pool(x)
            # 第一个卷积层
            c1 = self.activation(self.conv1(p1))
            c1 = self.dropout(c1)
            # 第二个卷积层
            c2 = self.activation(self.conv2(c1))
            c2 = self.dropout(c2)

            # 计算较小的长度
            min_length = min(c2.size(2), p1.size(2))
            if min_length <= 0:
                break

            # 截取较小长度的部分
            p1 = p1[:, :, :min_length]
            c2 = c2[:, :, :min_length]

            # 残差连接
            x = c2 + p1
            iteration += 1
            
        return x


class BiLSTMAttention(nn.Module):
    """双向LSTM + 注意力机制"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # 自注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 输出层
        output = self.fc(self.dropout(pooled))
        return output


class TCBAMModel(BaseModel):
    """TCBAM模型：Transformer + CNN + BiLSTM + Attention + CBAM"""
    
    def __init__(self, vocab_size, num_classes=2, embed_dim=128, hidden_dim=128, 
                 num_filters=128, num_heads=8, num_layers=2, dropout=0.1):
        super(TCBAMModel, self).__init__(vocab_size, num_classes)
        
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
        # DPCNN输出需要展平，BiLSTM输出是固定维度
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_filters * 4, num_filters * 2),  # 4个特征源：2个DPCNN + 2个BiLSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters * 2, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 嵌入 + 位置编码
        embedded = self.embedding(x)
        if seq_len <= self.pos_encoding.size(0):
            embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer编码
        # 创建padding mask
        padding_mask = (x == 0)
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # 转换为CNN输入格式 (batch, channels, seq_len)
        cnn_input = transformer_out.transpose(1, 2)
        
        # 分支1: kernel_size=2
        c11 = F.relu(self.conv1_k2(cnn_input))
        c11 = self.dropout(c11)
        
        # CBAM注意力
        cbam1_out = self.cbam1(c11)
        
        c12 = F.relu(self.conv2_k2(c11))
        c12 = self.dropout(c12)
        
        # 残差连接 + DPCNN (确保维度匹配)
        min_len = min(cbam1_out.size(2), c12.size(2))
        cbam1_out = cbam1_out[:, :, :min_len]
        c12 = c12[:, :, :min_len]
        residual1 = cbam1_out + c12
        dpcnn1_out = self.dpcnn1(residual1)
        
        # BiLSTM + Attention
        bilstm1_input = c12.transpose(1, 2)  # (batch, seq_len, channels)
        bilstm1_out = self.bilstm_att1(bilstm1_input)
        
        # 分支2: kernel_size=3
        c21 = F.relu(self.conv1_k3(cnn_input))
        c21 = self.dropout(c21)
        
        c22 = F.relu(self.conv2_k3(c21))
        c22 = self.dropout(c22)
        
        c23 = F.relu(self.conv3_k3(c22))
        c23 = self.dropout(c23)
        
        # CBAM注意力
        cbam2_out = self.cbam2(c23)
        
        # DPCNN
        dpcnn2_out = self.dpcnn2(cbam2_out)
        
        # BiLSTM + Attention
        bilstm2_input = c22.transpose(1, 2)  # (batch, seq_len, channels)
        bilstm2_out = self.bilstm_att2(bilstm2_input)
        
        # 特征融合
        # DPCNN输出需要全局平均池化
        dpcnn1_pooled = F.adaptive_avg_pool1d(dpcnn1_out, 1).squeeze(-1)
        dpcnn2_pooled = F.adaptive_avg_pool1d(dpcnn2_out, 1).squeeze(-1)
        
        # 拼接所有特征
        combined_features = torch.cat([
            dpcnn1_pooled,  # (batch, num_filters)
            bilstm1_out,    # (batch, num_filters)
            bilstm2_out,    # (batch, num_filters)
            dpcnn2_pooled   # (batch, num_filters)
        ], dim=1)
        
        # 特征融合
        fused_features = self.feature_fusion(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        print(f"📋 模型信息:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {model_size_mb:.2f} MB")
        print(f"  词汇表大小: {self.vocab_size}")
        print(f"  类别数: {self.num_classes}")
