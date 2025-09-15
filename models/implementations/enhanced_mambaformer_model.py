#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - Enhanced MambaFormer Model
基于MambaFormer的增强DGA检测模型，集成多种注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.base_model import BaseModel


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class ChannelSpatialAttention(nn.Module):
    """通道-空间注意力机制（增强版CBAM）"""
    
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        
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
            nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False),
            nn.Sigmoid()
        )
        
        # Global Context Attention
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels, 1, bias=False),
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
        
        # Global context attention
        global_att = self.global_context(x)
        x = x * global_att
        
        return x


class SelectiveScanMamba(nn.Module):
    """增强版Mamba选择性扫描模块"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand_factor * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space parameters
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, length, dim = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (batch, length, 2 * d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, length)
        x = self.conv1d(x)[:, :, :length]  # Causal convolution
        x = x.transpose(1, 2)  # (batch, length, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM
        y = self.selective_scan(x)
        
        # Residual connection
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        return self.dropout(output)
    
    def selective_scan(self, x):
        batch, length, d_inner = x.shape
        
        # Get SSM parameters
        x_proj_out = self.x_proj(x)  # (batch, length, d_state * 2)
        B, C = x_proj_out.split([self.d_state, self.d_state], dim=-1)
        
        # Delta (time step)
        delta = F.softplus(self.dt_proj(x))  # (batch, length, d_inner)
        
        # A matrix
        A = -torch.exp(self.A_log.float())  # (d_state,)
        
        # Selective scan
        y = self._selective_scan_fn(x, delta, A, B, C)
        
        return y
    
    def _selective_scan_fn(self, u, delta, A, B, C):
        batch, length, d_inner = u.shape
        d_state = A.shape[0]
        
        # Initialize state
        h = torch.zeros(batch, d_state, d_inner, device=u.device, dtype=u.dtype)
        
        outputs = []
        for i in range(length):
            # Current inputs
            u_i = u[:, i, :]  # (batch, d_inner)
            delta_i = delta[:, i, :]  # (batch, d_inner)
            B_i = B[:, i, :]  # (batch, d_state)
            C_i = C[:, i, :]  # (batch, d_state)
            
            # Discretize
            dA = torch.exp(delta_i.unsqueeze(1) * A.unsqueeze(0).unsqueeze(-1))  # (batch, d_state, d_inner)
            dB = delta_i.unsqueeze(1) * B_i.unsqueeze(-1)  # (batch, d_state, d_inner)
            
            # Update state
            h = h * dA + dB * u_i.unsqueeze(1)
            
            # Output
            y_i = torch.sum(h * C_i.unsqueeze(-1), dim=1)  # (batch, d_inner)
            outputs.append(y_i)
        
        y = torch.stack(outputs, dim=1)  # (batch, length, d_inner)
        
        # Add skip connection
        y = y + u * self.D
        
        return y


class EnhancedMambaFormerBlock(nn.Module):
    """增强版MambaFormer块"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Mamba component
        self.mamba = SelectiveScanMamba(d_model, d_state, d_conv, expand_factor, dropout)
        
        # Multi-head self-attention
        self.self_attention = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        # Mamba processing
        mamba_out = self.mamba(self.norm1(x))
        x = x + mamba_out
        
        # Self-attention processing
        attn_out, attn_weights = self.self_attention(self.norm2(x), self.norm2(x), self.norm2(x), mask)
        x = x + attn_out
        
        # Gated fusion of Mamba and Attention outputs
        combined = torch.cat([mamba_out, attn_out], dim=-1)
        gate_weights = self.gate(combined)
        fused = gate_weights * mamba_out + (1 - gate_weights) * attn_out
        x = x + fused
        
        # Feed-forward network
        ffn_out = self.ffn(self.norm3(x))
        x = x + ffn_out
        
        return x, attn_weights


class EnhancedMambaFormerModel(BaseModel):
    """增强版MambaFormer DGA检测模型"""
    
    def __init__(self, vocab_size, d_model=256, num_layers=6, d_state=16, d_conv=4, 
                 expand_factor=2, num_heads=8, num_classes=2, dropout=0.1, max_length=60):
        super(EnhancedMambaFormerModel, self).__init__(vocab_size, num_classes)
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Character-level CNN for local features
        self.char_cnn = nn.ModuleList([
            nn.Conv1d(d_model, d_model//2, kernel_size=k, padding=k//2)
            for k in [2, 3, 4, 5]
        ])
        self.char_pool = nn.AdaptiveMaxPool1d(1)
        
        # Channel-Spatial Attention
        self.cs_attention = ChannelSpatialAttention(d_model)
        
        # MambaFormer blocks
        self.blocks = nn.ModuleList([
            EnhancedMambaFormerBlock(d_model, d_state, d_conv, expand_factor, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion
        feature_dim = d_model * 3 + (d_model//2) * 4  # global + local features
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)  # (batch, seq_len, d_model)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # Character-level CNN features
        x_cnn = x.transpose(1, 2)  # (batch, d_model, seq_len)
        cnn_features = []
        for conv in self.char_cnn:
            conv_out = F.relu(conv(x_cnn))
            pooled = self.char_pool(conv_out).squeeze(-1)  # (batch, d_model//2)
            cnn_features.append(pooled)
        local_features = torch.cat(cnn_features, dim=1)  # (batch, d_model*2)
        
        # Channel-Spatial Attention
        x_cnn = self.cs_attention(x_cnn)
        x = x_cnn.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Create padding mask
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        
        # MambaFormer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, padding_mask)
            attention_weights.append(attn_weights)
        
        # Global feature aggregation
        x_transpose = x.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # Mask out padding tokens
        mask = (x.sum(dim=-1) != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
        x_masked = x * mask
        
        # Global pooling
        global_avg = x_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (batch, d_model)
        global_max = torch.max(x_masked + (1 - mask) * (-1e8), dim=1)[0]  # (batch, d_model)
        
        # Last token representation (for sequence classification)
        last_token = x[:, -1, :]  # (batch, d_model)
        
        # Feature fusion
        global_features = torch.cat([global_avg, global_max, last_token], dim=1)  # (batch, d_model*3)
        all_features = torch.cat([global_features, local_features], dim=1)  # (batch, feature_dim)
        
        fused_features = self.feature_fusion(all_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_attention_weights(self, x):
        """获取注意力权重用于可视化"""
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # Character-level CNN
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cs_attention(x_cnn)
        x = x_cnn.transpose(1, 2)
        
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, padding_mask)
            attention_weights.append(attn_weights)
        
        return attention_weights
    
    def get_feature_importance(self, x):
        """获取特征重要性"""
        # Enable gradient computation
        x.requires_grad_()
        
        # Forward pass
        logits = self.forward(x)
        
        # Compute gradients
        grad_outputs = torch.ones_like(logits)
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Feature importance as gradient magnitude
        importance = torch.abs(gradients).mean(dim=0)
        
        return importance