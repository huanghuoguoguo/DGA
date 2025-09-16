#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级孪生网络增强模型
简化版本，专注于核心功能验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from .siamese_addon import SiameseHead, CosineContrastiveLoss


class SiameseMoEModel(BaseModel):
    """轻量级孪生网络增强模型"""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_experts=4, 
                 top_k=2, num_classes=2, dropout=0.1, max_length=40,
                 siamese_emb_dim=128, load_balance_weight=0.01):
        super(SiameseMoEModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.siamese_emb_dim = siamese_emb_dim
        
        # 简化的共享特征提取器 - 使用BiLSTM
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 双向LSTM作为特征提取器
        self.feature_extractor = nn.LSTM(
            d_model, d_model // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 主分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 孪生网络头部
        self.siamese_head = SiameseHead(
            in_dim=d_model,
            emb_dim=siamese_emb_dim,
            dropout=dropout
        )
        
        # 孪生损失函数
        self.siamese_loss_fn = CosineContrastiveLoss(margin=0.4)
        
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def extract_features(self, x):
        """提取共享特征表示"""
        # 嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        embedded = self.dropout(embedded)
        
        # 创建mask处理padding
        mask = (x != 0).float()
        lengths = mask.sum(dim=1).long()
        
        # 打包序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM特征提取
        packed_output, (hidden, _) = self.feature_extractor(packed_embedded)
        
        # 解包
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 使用最后一个有效时间步的输出
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            if lengths[i] > 0:
                last_outputs.append(lstm_output[i, lengths[i]-1, :])
            else:
                last_outputs.append(lstm_output[i, 0, :])  # 处理空序列
        
        features = torch.stack(last_outputs)
        
        # 归一化特征
        features = self.norm(features)
        
        return features
    
    def forward(self, x):
        """主分类前向传播"""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def forward_siamese(self, x1, x2):
        """孪生网络前向传播"""
        # 提取两个输入的特征
        features1 = self.extract_features(x1)
        features2 = self.extract_features(x2)
        
        # 通过孪生头部获取嵌入向量
        z1 = self.siamese_head(features1)
        z2 = self.siamese_head(features2)
        
        return z1, z2
    
    def forward_joint(self, x1, x2, same_family):
        """联合训练前向传播"""
        # 孪生分支
        z1, z2 = self.forward_siamese(x1, x2)
        siamese_loss = self.siamese_loss_fn(z1, z2, same_family)
        
        # 主分类分支 (使用x1)
        logits = self.forward(x1)
        
        return {
            'logits': logits,
            'z1': z1,
            'z2': z2,
            'siamese_loss': siamese_loss
        }
    
    def compute_similarity(self, x1, x2):
        """计算两个样本的相似度"""
        with torch.no_grad():
            z1, z2 = self.forward_siamese(x1, x2)
            similarity = F.cosine_similarity(z1, z2)
        return similarity
    
    def get_embedding(self, x):
        """获取样本的嵌入表示"""
        with torch.no_grad():
            features = self.extract_features(x)
            embedding = self.siamese_head(features)
        return embedding