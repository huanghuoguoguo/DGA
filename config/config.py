#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 配置管理
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 5
    random_seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = './data/processed/small_dga_dataset.pkl'
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    max_length: int = 40


@dataclass
class ModelConfig:
    """模型配置"""
    d_model: int = 128
    dropout: float = 0.1
    num_classes: int = 2


class Config:
    """统一配置管理"""
    
    def __init__(self):
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        
        # 路径配置
        self.paths = {
            'data_dir': './data',
            'processed_data_dir': './data/processed',
            'models_dir': './data/models',
            'results_dir': './data/results',
            'logs_dir': './logs'
        }
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要目录存在"""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def get_model_save_path(self, model_name: str) -> str:
        """获取模型保存路径"""
        return os.path.join(self.paths['models_dir'], f'best_{model_name.lower().replace(" ", "_")}_model.pth')
    
    def get_results_save_path(self, model_name: str) -> str:
        """获取结果保存路径"""
        return os.path.join(self.paths['results_dir'], f'{model_name.lower().replace(" ", "_")}_results.pkl')


# 全局配置实例
config = Config()