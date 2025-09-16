#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAN增强的孪生网络模型
实现生成-度量联合框架，用于DGA检测的小样本学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from .siamese_addon import SiameseHead, CosineContrastiveLoss


class DGAGenerator(nn.Module):
    """DGA域名生成器"""
    
    def __init__(self, vocab_size, max_length=40, latent_dim=100, hidden_dim=256):
        super(DGAGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 噪声到隐藏状态的映射
        self.noise_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM生成器
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, batch_size, device='cpu'):
        """生成DGA域名序列"""
        # 生成随机噪声
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        
        # 噪声到初始隐藏状态
        hidden_input = self.noise_to_hidden(noise)  # [batch_size, hidden_dim]
        
        # 初始化LSTM状态
        h0 = hidden_input.unsqueeze(0)  # [1, batch_size, hidden_dim]
        c0 = torch.zeros_like(h0)
        
        # 生成序列
        outputs = []
        input_token = hidden_input.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        for _ in range(self.max_length):
            lstm_out, (h0, c0) = self.lstm(input_token, (h0, c0))
            token_logits = self.output_layer(lstm_out.squeeze(1))  # [batch_size, vocab_size]
            
            # 使用Gumbel-Softmax进行可微分采样
            token_probs = F.gumbel_softmax(token_logits, tau=1.0, hard=True)
            outputs.append(token_probs)
            
            # 下一个输入
            input_token = lstm_out
        
        # 拼接输出序列
        generated_sequence = torch.stack(outputs, dim=1)  # [batch_size, max_length, vocab_size]
        
        # 转换为token索引
        generated_tokens = torch.argmax(generated_sequence, dim=-1)  # [batch_size, max_length]
        
        return generated_tokens, generated_sequence


class DGADiscriminator(nn.Module):
    """DGA域名判别器"""
    
    def __init__(self, vocab_size, d_model=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super(DGADiscriminator, self).__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # BiLSTM特征提取
        self.lstm = nn.LSTM(
            d_model, hidden_dim // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 判别器头部
        self.discriminator_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """判别真实/生成的域名"""
        # 嵌入
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # 创建mask处理padding
        mask = (x != 0).float()
        lengths = mask.sum(dim=1).long()
        
        # 打包序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM特征提取
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        
        # 解包
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 使用最后一个有效时间步的输出
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            if lengths[i] > 0:
                last_outputs.append(lstm_output[i, lengths[i]-1, :])
            else:
                last_outputs.append(lstm_output[i, 0, :])
        
        last_hidden = torch.stack(last_outputs)  # [batch_size, hidden_dim]
        
        # 判别
        discriminator_score = self.discriminator_head(last_hidden)
        
        return discriminator_score.squeeze(-1), last_hidden


class HardNegativeMiner:
    """难负例挖掘器"""
    
    def __init__(self, generator, discriminator, siamese_model, vocab_size):
        self.generator = generator
        self.discriminator = discriminator
        self.siamese_model = siamese_model
        self.vocab_size = vocab_size
    
    def mine_hard_negatives(self, anchor_samples, num_negatives=32, difficulty_threshold=0.7):
        """挖掘难负例"""
        device = anchor_samples.device
        batch_size = anchor_samples.size(0)
        
        hard_negatives = []
        
        for _ in range(num_negatives // batch_size + 1):
            # 生成候选负例
            generated_tokens, _ = self.generator(batch_size, device)
            
            # 计算与锚点的相似度
            anchor_embeddings = self.siamese_model.get_embedding(anchor_samples)
            negative_embeddings = self.siamese_model.get_embedding(generated_tokens)
            
            similarities = F.cosine_similarity(
                anchor_embeddings.unsqueeze(1), 
                negative_embeddings.unsqueeze(0), 
                dim=-1
            )
            
            # 选择难负例（相似度高但不同类别）
            max_similarities, _ = similarities.max(dim=0)
            hard_mask = max_similarities > difficulty_threshold
            
            if hard_mask.sum() > 0:
                hard_negatives.append(generated_tokens[hard_mask])
        
        if hard_negatives:
            return torch.cat(hard_negatives, dim=0)[:num_negatives]
        else:
            # 如果没有找到难负例，返回随机生成的负例
            generated_tokens, _ = self.generator(num_negatives, device)
            return generated_tokens


class GANSiameseModel(BaseModel):
    """GAN增强的孪生网络模型"""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_classes=2, 
                 dropout=0.1, siamese_emb_dim=128, latent_dim=100, 
                 discriminator_hidden=256, generator_hidden=256):
        super(GANSiameseModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.siamese_emb_dim = siamese_emb_dim
        
        # 孪生网络组件（复用之前的设计）
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        self.feature_extractor = nn.LSTM(
            d_model, d_model // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.siamese_head = SiameseHead(
            in_dim=d_model,
            emb_dim=siamese_emb_dim,
            dropout=dropout
        )
        
        # GAN组件
        self.generator = DGAGenerator(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            hidden_dim=generator_hidden
        )
        
        self.discriminator = DGADiscriminator(
            vocab_size=vocab_size,
            d_model=d_model,
            hidden_dim=discriminator_hidden,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 难负例挖掘器
        self.hard_negative_miner = HardNegativeMiner(
            self.generator, self.discriminator, self, vocab_size
        )
        
        # 损失函数
        self.siamese_loss_fn = CosineContrastiveLoss(margin=0.4)
        self.adversarial_loss_fn = nn.BCELoss()
        
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
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
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        mask = (x != 0).float()
        lengths = mask.sum(dim=1).long()
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, _) = self.feature_extractor(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            if lengths[i] > 0:
                last_outputs.append(lstm_output[i, lengths[i]-1, :])
            else:
                last_outputs.append(lstm_output[i, 0, :])
        
        features = torch.stack(last_outputs)
        features = self.norm(features)
        
        return features
    
    def forward(self, x):
        """主分类前向传播"""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def forward_siamese(self, x1, x2):
        """孪生网络前向传播"""
        features1 = self.extract_features(x1)
        features2 = self.extract_features(x2)
        
        z1 = self.siamese_head(features1)
        z2 = self.siamese_head(features2)
        
        return z1, z2
    
    def forward_gan_joint(self, x1, x2, same_family, use_hard_negatives=True):
        """GAN增强的联合训练前向传播"""
        device = x1.device
        batch_size = x1.size(0)
        
        # 孪生分支
        z1, z2 = self.forward_siamese(x1, x2)
        siamese_loss = self.siamese_loss_fn(z1, z2, same_family)
        
        # 主分类分支
        logits = self.forward(x1)
        
        # GAN分支
        # 1. 生成器生成假样本
        generated_tokens, generated_sequence = self.generator(batch_size, device)
        
        # 2. 判别器判别真假
        real_scores, _ = self.discriminator(x1)
        fake_scores, _ = self.discriminator(generated_tokens.detach())  # detach避免梯度传播到生成器
        
        # 3. 对抗损失
        real_labels = torch.ones(real_scores.size(0), device=device)
        fake_labels = torch.zeros(fake_scores.size(0), device=device)
        
        d_loss_real = self.adversarial_loss_fn(real_scores, real_labels)
        d_loss_fake = self.adversarial_loss_fn(fake_scores, fake_labels)
        discriminator_loss = (d_loss_real + d_loss_fake) / 2
        
        # 生成器损失（重新计算fake_scores以保持梯度）
        fake_scores_for_g, _ = self.discriminator(generated_tokens)
        generator_loss = self.adversarial_loss_fn(fake_scores_for_g, real_labels[:fake_scores_for_g.size(0)])
        
        # 4. 难负例增强（简化版本，避免复杂的梯度计算）
        hard_negative_loss = torch.tensor(0.0, device=device)
        if use_hard_negatives and batch_size >= 4:
            try:
                # 简化的难负例：使用batch内的负样本
                anchor_embeddings = self.siamese_head(self.extract_features(x1[:2]))
                negative_embeddings = self.siamese_head(self.extract_features(x1[2:4]))
                
                # 计算相似度并应用margin loss
                similarities = F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=-1)
                hard_negative_loss = F.relu(similarities - 0.3).mean()  # margin = 0.3
            except Exception as e:
                hard_negative_loss = torch.tensor(0.0, device=device)
        
        return {
            'logits': logits,
            'z1': z1.detach(),  # detach避免重复计算梯度
            'z2': z2.detach(),
            'siamese_loss': siamese_loss,
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
            'hard_negative_loss': hard_negative_loss,
            'generated_tokens': generated_tokens.detach()
        }
    
    def get_embedding(self, x):
        """获取样本的嵌入表示"""
        with torch.no_grad():
            features = self.extract_features(x)
            embedding = self.siamese_head(features)
        return embedding
    
    def generate_hard_negatives(self, anchor_samples, num_negatives=32):
        """生成难负例"""
        return self.hard_negative_miner.mine_hard_negatives(anchor_samples, num_negatives)