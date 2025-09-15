#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 专门化MoE模型
使用针对字符级和字典级DGA的专门化专家，以及高级注意力模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from .specialized_experts import (
    CharacterLevelExpert, DictionaryLevelExpert, 
    BiGRUAttentionExpert, CNNWithCBAMExpert, TransformerExpert
)


class IntelligentGateNetwork(nn.Module):
    """智能门控网络 - 基于DGA特征的专家选择"""
    
    def __init__(self, vocab_size, d_model=128, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        
        # 基础特征提取
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.feature_extractor = nn.LSTM(d_model, 32, batch_first=True, bidirectional=True)
        
        # DGA特征分析器
        self.entropy_analyzer = nn.Linear(1, 8)
        self.length_analyzer = nn.Linear(1, 8)
        self.char_dist_analyzer = nn.Linear(vocab_size, 16)
        self.pattern_analyzer = nn.Linear(4, 8)  # 重复、递增、递减、周期性
        
        # 门控决策网络
        feature_dim = 64 + 8 + 8 + 16 + 8  # lstm + entropy + length + char_dist + pattern
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def analyze_entropy(self, x):
        """分析序列熵"""
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
        
        entropy_tensor = torch.tensor(entropies, device=x.device).unsqueeze(1)
        return self.entropy_analyzer(entropy_tensor / 4.0)  # 归一化
    
    def analyze_length(self, x):
        """分析序列长度"""
        lengths = (x != 0).sum(dim=1).float().unsqueeze(1)
        return self.length_analyzer(lengths / 40.0)  # 归一化
    
    def analyze_char_distribution(self, x):
        """分析字符分布"""
        batch_size = x.size(0)
        char_dist = torch.zeros(batch_size, self.vocab_size, device=x.device)
        
        for i in range(batch_size):
            seq = x[i]
            valid_seq = seq[seq != 0]
            if len(valid_seq) > 0:
                for char_idx in valid_seq:
                    char_dist[i, char_idx] += 1
                char_dist[i] = char_dist[i] / len(valid_seq)
        
        return self.char_dist_analyzer(char_dist)
    
    def analyze_patterns(self, x):
        """分析序列模式"""
        batch_size = x.size(0)
        pattern_features = []
        
        for i in range(batch_size):
            seq = x[i]
            valid_seq = seq[seq != 0]
            
            if len(valid_seq) <= 1:
                features = torch.zeros(4)
            else:
                # 重复字符比例
                repeat_count = sum(1 for j in range(len(valid_seq)-1) if valid_seq[j] == valid_seq[j+1])
                repeat_ratio = repeat_count / (len(valid_seq) - 1)
                
                # 递增/递减模式
                inc_count = sum(1 for j in range(len(valid_seq)-1) if valid_seq[j+1] > valid_seq[j])
                dec_count = sum(1 for j in range(len(valid_seq)-1) if valid_seq[j+1] < valid_seq[j])
                inc_ratio = inc_count / (len(valid_seq) - 1)
                dec_ratio = dec_count / (len(valid_seq) - 1)
                
                # 周期性（简单检测）
                periodicity = 0
                if len(valid_seq) >= 4:
                    period2_matches = sum(1 for j in range(len(valid_seq)-2) if valid_seq[j] == valid_seq[j+2])
                    periodicity = period2_matches / max(1, len(valid_seq) - 2)
                
                features = torch.tensor([repeat_ratio, inc_ratio, dec_ratio, periodicity])
            
            pattern_features.append(features)
        
        pattern_tensor = torch.stack(pattern_features).to(x.device)
        return self.pattern_analyzer(pattern_tensor)
    
    def forward(self, x):
        # 基础序列特征
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.feature_extractor(embedded)
        seq_features = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 64)
        
        # DGA特征分析
        entropy_features = self.analyze_entropy(x)  # (batch, 8)
        length_features = self.analyze_length(x)    # (batch, 8)
        char_dist_features = self.analyze_char_distribution(x)  # (batch, 16)
        pattern_features = self.analyze_patterns(x)  # (batch, 8)
        
        # 特征融合
        all_features = torch.cat([
            seq_features, entropy_features, length_features,
            char_dist_features, pattern_features
        ], dim=1)
        
        # 门控权重计算
        gate_weights = self.gate_network(all_features)
        
        return gate_weights


class SpecializedMoEModel(BaseModel):
    """专门化MoE模型 - 针对字符级和字典级DGA的专家组合"""
    
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1, expert_config="char_dict"):
        super(SpecializedMoEModel, self).__init__(vocab_size, num_classes)
        
        self.expert_config = expert_config
        self.d_model = d_model
        
        # 根据配置选择专家组合
        if expert_config == "char_dict":
            # 字符级 + 字典级专家
            self.experts = nn.ModuleList([
                self._create_char_expert(vocab_size, d_model, dropout),
                self._create_dict_expert(vocab_size, d_model, dropout)
            ])
            self.expert_names = ["character_level", "dictionary_level"]
            self.num_experts = 2
            
        elif expert_config == "advanced":
            # 高级模型组合
            self.experts = nn.ModuleList([
                self._create_bigru_expert(vocab_size, d_model, dropout),
                self._create_cnn_cbam_expert(vocab_size, d_model, dropout),
                self._create_transformer_expert(vocab_size, d_model, dropout)
            ])
            self.expert_names = ["bigru_attention", "cnn_cbam", "transformer"]
            self.num_experts = 3
            
        elif expert_config == "hybrid":
            # 混合专家：专门化 + 高级模型
            self.experts = nn.ModuleList([
                self._create_char_expert(vocab_size, d_model, dropout),
                self._create_dict_expert(vocab_size, d_model, dropout),
                self._create_bigru_expert(vocab_size, d_model, dropout),
                self._create_cnn_cbam_expert(vocab_size, d_model, dropout)
            ])
            self.expert_names = ["character_level", "dictionary_level", "bigru_attention", "cnn_cbam"]
            self.num_experts = 4
        
        # 智能门控网络
        self.gate = IntelligentGateNetwork(vocab_size, d_model, self.num_experts)
        
        # 最终分类器
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 损失权重
        self.load_balance_weight = 0.3
        self.diversity_weight = 0.2
    
    def _create_char_expert(self, vocab_size, d_model, dropout):
        """创建字符级专家（去掉分类层）"""
        class CharExpertWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = CharacterLevelExpert(vocab_size, d_model, 2, dropout)
                # 移除分类层，只保留特征提取
                self.feature_extractor = nn.Sequential(
                    *list(self.expert.children())[:-1]  # 除了最后的分类器
                )
                self.output_proj = nn.Linear(d_model // 2, d_model)
            
            def forward(self, x):
                # 复制CharacterLevelExpert的forward逻辑，但不包括最终分类
                embedded = self.expert.embedding(x)
                embedded = self.expert.dropout(embedded)
                embedded = embedded.transpose(1, 2)
                
                conv_outputs = []
                for conv, bn in zip(self.expert.conv_layers, self.expert.batch_norms):
                    conv_out = F.relu(bn(conv(embedded)))
                    pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
                    conv_outputs.append(pooled.squeeze(2))
                
                conv_features = torch.cat(conv_outputs, dim=1)
                conv_features_expanded = conv_features.unsqueeze(2)
                attended_features = self.expert.cbam(conv_features_expanded).squeeze(2)
                
                freq_features = self.expert.compute_char_frequency(x)
                entropy_features = self.expert.compute_entropy(x) * self.expert.entropy_weight
                
                all_features = torch.cat([attended_features, freq_features, entropy_features], dim=1)
                fused_features = self.expert.feature_fusion(all_features)
                
                return self.output_proj(fused_features)
        
        return CharExpertWrapper()
    
    def _create_dict_expert(self, vocab_size, d_model, dropout):
        """创建字典级专家（去掉分类层）"""
        class DictExpertWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = DictionaryLevelExpert(vocab_size, d_model, 2, dropout)
                self.output_proj = nn.Linear(d_model // 2, d_model)
            
            def forward(self, x):
                batch_size, seq_len = x.shape
                
                embedded = self.expert.embedding(x)
                embedded = self.expert.dropout(embedded)
                
                gru_out, _ = self.expert.bigru(embedded)
                pos_enc = self.expert.pos_encoding[:, :seq_len, :].to(x.device)
                gru_out = gru_out + pos_enc
                
                attn_out, _ = self.expert.multihead_attention(gru_out, gru_out, gru_out)
                
                padding_mask = (x == 0)
                mask = (~padding_mask).float().unsqueeze(-1)
                pooled_attn = (attn_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                
                similarity_features = self.expert.word_similarity(pooled_attn)
                semantic_features = self.expert.semantic_analyzer(pooled_attn)
                length_features = self.expert.compute_length_features(x)
                
                all_features = torch.cat([pooled_attn, similarity_features, semantic_features, length_features], dim=1)
                fused_features = self.expert.feature_fusion(all_features)
                
                return self.output_proj(fused_features)
        
        return DictExpertWrapper()
    
    def _create_bigru_expert(self, vocab_size, d_model, dropout):
        """创建BiGRU+Attention专家（去掉分类层）"""
        class BiGRUExpertWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = BiGRUAttentionExpert(vocab_size, d_model, 2, dropout)
                self.output_proj = nn.Linear(128, d_model)
            
            def forward(self, x):
                embedded = self.expert.embedding(x)
                embedded = self.expert.dropout(embedded)
                
                gru_out, _ = self.expert.bigru(embedded)
                attention_weights = self.expert.attention(gru_out)
                attention_weights = F.softmax(attention_weights, dim=1)
                weighted_output = torch.sum(gru_out * attention_weights, dim=1)
                
                return self.output_proj(weighted_output)
        
        return BiGRUExpertWrapper()
    
    def _create_cnn_cbam_expert(self, vocab_size, d_model, dropout):
        """创建CNN+CBAM专家（去掉分类层）"""
        class CNNCBAMExpertWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = CNNWithCBAMExpert(vocab_size, d_model, 2, dropout)
                self.output_proj = nn.Linear(128 * 4, d_model)
            
            def forward(self, x):
                embedded = self.expert.embedding(x)
                embedded = self.expert.dropout(embedded)
                embedded = embedded.transpose(1, 2)
                
                conv_outputs = []
                for conv, bn in zip(self.expert.conv_layers, self.expert.batch_norms):
                    conv_out = F.relu(bn(conv(embedded)))
                    pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
                    conv_outputs.append(pooled.squeeze(2))
                
                features = torch.cat(conv_outputs, dim=1)
                features_expanded = features.unsqueeze(2)
                attended_features = self.expert.cbam(features_expanded).squeeze(2)
                
                return self.output_proj(attended_features)
        
        return CNNCBAMExpertWrapper()
    
    def _create_transformer_expert(self, vocab_size, d_model, dropout):
        """创建Transformer专家（去掉分类层）"""
        class TransformerExpertWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = TransformerExpert(vocab_size, d_model, 2, dropout)
                self.output_proj = nn.Linear(d_model // 2, d_model)
            
            def forward(self, x):
                embedded = self.expert.embedding(x) * math.sqrt(self.expert.d_model)
                embedded = embedded + self.expert.pos_encoding[:, :x.size(1), :].to(x.device)
                embedded = self.expert.dropout(embedded)
                
                padding_mask = (x == 0)
                transformer_out = self.expert.transformer(embedded, src_key_padding_mask=padding_mask)
                
                mask = (~padding_mask).float().unsqueeze(-1)
                pooled = (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                
                # 只通过第一层分类器（特征提取）
                features = self.expert.classifier[0](pooled)  # Linear layer
                features = self.expert.classifier[1](features)  # ReLU
                features = self.expert.classifier[2](features)  # Dropout
                
                return self.output_proj(features)
        
        return TransformerExpertWrapper()
    
    def compute_load_balance_loss(self, gate_weights):
        """计算负载均衡损失"""
        expert_usage = torch.mean(gate_weights, dim=0)
        target_usage = 1.0 / self.num_experts
        load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, target_usage))
        return load_balance_loss
    
    def compute_diversity_loss(self, expert_outputs):
        """计算专家输出多样性损失"""
        if len(expert_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)
        
        similarities = []
        for i in range(len(expert_outputs)):
            for j in range(i+1, len(expert_outputs)):
                sim = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=1).mean()
                similarities.append(sim)
        
        if similarities:
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
            expert_name = self.expert_names[i]
            expert_outputs[expert_name] = expert(x)
        return expert_outputs
    
    def compute_total_loss(self, logits, targets, gate_weights, expert_outputs):
        """计算总损失"""
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