#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - Homogeneous MambaFormer MoE Model
同构MambaFormer混合专家模型，用于对比异构vs同构专家的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel
from .enhanced_mambaformer_model import EnhancedMambaFormerModel


class HomogeneousGatingNetwork(nn.Module):
    """同构专家的门控网络 - 基于序列特征的智能路由"""
    
    def __init__(self, vocab_size, d_model=128, num_experts=4, max_length=60):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 基础特征提取
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.feature_extractor = nn.LSTM(d_model, 64, batch_first=True, bidirectional=True)
        
        # 序列特征分析器
        self.entropy_analyzer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.length_analyzer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.char_dist_analyzer = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(4, 16),  # 模式特征
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 特征融合
        feature_dim = 128 + 8 + 8 + 16 + 8  # 总特征维度 = 168
        # 确保特征维度能被注意力头数整除
        adjusted_feature_dim = ((feature_dim + 7) // 8) * 8  # 调整为8的倍数 = 168
        self.feature_proj = nn.Linear(feature_dim, adjusted_feature_dim)
        self.feature_attention = nn.MultiheadAttention(adjusted_feature_dim, num_heads=8, batch_first=True)
        
        # 门控决策网络
        self.gate_network = nn.Sequential(
            nn.Linear(adjusted_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )
        
        # 温度参数用于Gumbel Softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
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
        """分析序列长度特征"""
        lengths = (x != 0).sum(dim=1).float().unsqueeze(1)
        return self.length_analyzer(lengths / 60.0)  # 归一化
    
    def analyze_char_distribution(self, x):
        """分析字符分布特征"""
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
        """分析序列模式特征"""
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
                inc_ratio = inc_count / (len(valid_seq) - 1)
                
                # 字符多样性
                unique_chars = len(torch.unique(valid_seq))
                diversity = unique_chars / len(valid_seq)
                
                # 周期性检测
                periodicity = 0
                if len(valid_seq) >= 4:
                    period2_matches = sum(1 for j in range(len(valid_seq)-2) if valid_seq[j] == valid_seq[j+2])
                    periodicity = period2_matches / max(1, len(valid_seq) - 2)
                
                features = torch.tensor([repeat_ratio, inc_ratio, diversity, periodicity])
            
            pattern_features.append(features)
        
        pattern_tensor = torch.stack(pattern_features).to(x.device)
        return self.pattern_analyzer(pattern_tensor)
    
    def forward(self, x, training=True):
        # 基础序列特征
        embedded = self.embedding(x)
        lstm_out, (h_n, _) = self.feature_extractor(embedded)
        seq_features = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 128)
        
        # 多维特征分析
        entropy_features = self.analyze_entropy(x)  # (batch, 8)
        length_features = self.analyze_length(x)    # (batch, 8)
        char_dist_features = self.analyze_char_distribution(x)  # (batch, 16)
        pattern_features = self.analyze_patterns(x)  # (batch, 8)
        
        # 特征融合
        all_features = torch.cat([
            seq_features, entropy_features, length_features,
            char_dist_features, pattern_features
        ], dim=1)  # (batch, 168)
        
        # 特征维度调整和自注意力特征融合
        all_features = self.feature_proj(all_features)  # (batch, adjusted_feature_dim)
        all_features = all_features.unsqueeze(1)  # (batch, 1, adjusted_feature_dim)
        attended_features, _ = self.feature_attention(all_features, all_features, all_features)
        attended_features = attended_features.squeeze(1)  # (batch, adjusted_feature_dim)
        
        # 门控权重计算
        gate_logits = self.gate_network(attended_features)
        
        if training:
            # 训练时使用Gumbel Softmax
            gate_weights = F.gumbel_softmax(gate_logits, tau=self.temperature, hard=False)
        else:
            # 推理时使用标准softmax
            gate_weights = F.softmax(gate_logits, dim=-1)
        
        return gate_weights, {
            'entropy': entropy_features,
            'length': length_features,
            'char_dist': char_dist_features,
            'patterns': pattern_features,
            'gate_logits': gate_logits
        }


class HomogeneousMambaFormerExpert(nn.Module):
    """同构MambaFormer专家 - 使用相同架构但不同初始化的专家"""
    
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_classes=2, dropout=0.1, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        
        # 使用增强版MambaFormer作为专家基础架构
        self.expert = EnhancedMambaFormerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # 移除分类层，只保留特征提取
        self.feature_dim = d_model
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 为每个专家添加独特的初始化
        self._init_expert_weights(expert_id)
        
    def _init_expert_weights(self, expert_id):
        """为每个专家使用不同的初始化策略"""
        # 设置不同的随机种子
        torch.manual_seed(42 + expert_id * 1000)
        
        for module in self.expert.modules():
            if isinstance(module, nn.Linear):
                if expert_id == 0:
                    # 专家0：Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                elif expert_id == 1:
                    # 专家1：Kaiming初始化
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif expert_id == 2:
                    # 专家2：正态分布初始化
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                else:
                    # 专家3+：均匀分布初始化
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02 * (1 + expert_id * 0.1))
        
        # 恢复全局随机种子
        torch.manual_seed(torch.initial_seed())
        
    def forward(self, x):
        # 获取MambaFormer的特征表示
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.expert.token_embedding(x)
        pos_emb = self.expert.position_embedding(positions)
        x_emb = token_emb + pos_emb
        
        # Character-level CNN features
        x_cnn = x_emb.transpose(1, 2)
        cnn_features = []
        for conv in self.expert.char_cnn:
            conv_out = F.relu(conv(x_cnn))
            pooled = self.expert.char_pool(conv_out).squeeze(-1)
            cnn_features.append(pooled)
        local_features = torch.cat(cnn_features, dim=1)
        
        # Channel-Spatial Attention
        x_cnn = self.expert.cs_attention(x_cnn)
        x_emb = x_cnn.transpose(1, 2)
        
        # MambaFormer blocks
        padding_mask = (x_emb.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        for block in self.expert.blocks:
            x_emb, _ = block(x_emb, padding_mask)
        
        # Global feature aggregation
        mask = (x_emb.sum(dim=-1) != 0).float().unsqueeze(-1)
        x_masked = x_emb * mask
        global_avg = x_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Feature fusion
        global_features = torch.cat([global_avg, global_avg, global_avg], dim=1)  # 简化版
        all_features = torch.cat([global_features, local_features], dim=1)
        
        # 通过特征融合层
        fused_features = self.expert.feature_fusion(all_features)
        
        return self.output_proj(fused_features)


class HomogeneousMambaFormerMoE(BaseModel):
    """同构MambaFormer MoE模型 - 使用相同架构的多个专家"""
    
    def __init__(self, vocab_size, d_model=256, num_classes=2, dropout=0.1, 
                 num_experts=4, load_balance_weight=0.3, diversity_weight=0.2):
        super(HomogeneousMambaFormerMoE, self).__init__(vocab_size, num_classes)
        
        self.num_experts = num_experts
        self.d_model = d_model
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        
        # 创建多个同构的MambaFormer专家
        self.experts = nn.ModuleList([
            HomogeneousMambaFormerExpert(
                vocab_size=vocab_size, 
                d_model=d_model, 
                num_layers=4,  # 所有专家使用相同的层数
                dropout=dropout,
                expert_id=i
            )
            for i in range(num_experts)
        ])
        
        # 专家名称
        self.expert_names = [f"mambaformer_expert_{i}" for i in range(num_experts)]
        
        # 门控网络
        self.gate = HomogeneousGatingNetwork(vocab_size, d_model//2, num_experts)
        
        # 专家输出融合
        self.expert_fusion = nn.Sequential(
            nn.Linear(d_model * num_experts, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 辅助损失权重
        self.aux_loss_weight = 0.1
        
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
    
    def forward(self, x, return_gate_weights=False, return_expert_outputs=False, return_aux_info=False):
        # 门控网络计算专家权重
        gate_weights, aux_info = self.gate(x, training=self.training)
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # (batch, d_model)
            expert_outputs.append(expert_output)
        
        # 加权混合专家输出
        weighted_outputs = []
        for i, expert_output in enumerate(expert_outputs):
            weighted_output = gate_weights[:, i:i+1] * expert_output
            weighted_outputs.append(weighted_output)
        
        # 专家输出融合
        all_expert_outputs = torch.cat(expert_outputs, dim=1)  # (batch, d_model * num_experts)
        fused_features = self.expert_fusion(all_expert_outputs)
        
        # 门控加权的最终输出
        mixed_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            mixed_output += gate_weights[:, i:i+1] * expert_output
        
        # 结合融合特征和加权输出
        final_features = fused_features + mixed_output
        
        # 最终分类
        logits = self.classifier(final_features)
        
        # 返回不同的信息
        result = [logits]
        if return_gate_weights:
            result.append(gate_weights)
        if return_expert_outputs:
            result.append(expert_outputs)
        if return_aux_info:
            result.append(aux_info)
        
        return result[0] if len(result) == 1 else tuple(result)
    
    def compute_total_loss(self, logits, targets, gate_weights, expert_outputs, aux_info=None):
        """计算总损失"""
        # 分类损失
        classification_loss = F.cross_entropy(logits, targets)
        
        # 负载均衡损失
        load_balance_loss = self.compute_load_balance_loss(gate_weights)
        
        # 多样性损失
        diversity_loss = self.compute_diversity_loss(expert_outputs)
        
        # 辅助损失（门控网络的正则化）
        aux_loss = 0.0
        if aux_info is not None and 'gate_logits' in aux_info:
            # 门控logits的熵正则化，鼓励决策的确定性
            gate_entropy = -torch.sum(F.softmax(aux_info['gate_logits'], dim=-1) * 
                                    F.log_softmax(aux_info['gate_logits'], dim=-1), dim=-1).mean()
            aux_loss = gate_entropy * self.aux_loss_weight
        
        # 总损失
        total_loss = (classification_loss + 
                     self.load_balance_weight * load_balance_loss + 
                     self.diversity_weight * diversity_loss + 
                     aux_loss)
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'load_balance_loss': load_balance_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'aux_loss': aux_loss if isinstance(aux_loss, float) else aux_loss.item(),
            'gate_entropy': gate_entropy.item() if aux_info and 'gate_logits' in aux_info else 0.0
        }
    
    def get_expert_usage_stats(self, dataloader):
        """获取专家使用统计"""
        self.eval()
        expert_usage = torch.zeros(self.num_experts)
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(next(self.parameters()).device)
                gate_weights, _ = self.gate(batch_x, training=False)
                
                # 统计每个样本使用的主要专家
                dominant_experts = torch.argmax(gate_weights, dim=1)
                for expert_idx in dominant_experts:
                    expert_usage[expert_idx] += 1
                
                total_samples += batch_x.size(0)
        
        expert_usage_ratio = expert_usage / total_samples
        
        return {
            'expert_usage_counts': expert_usage.tolist(),
            'expert_usage_ratios': expert_usage_ratio.tolist(),
            'expert_names': self.expert_names,
            'total_samples': total_samples
        }