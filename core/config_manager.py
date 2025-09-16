#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
负责加载和验证TOML配置文件
"""

import toml
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
    
    def load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典或None（如果加载失败）
        """
        try:
            if not os.path.exists(config_path):
                self.logger.error(f"配置文件不存在: {config_path}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
            
            # 验证配置
            if not self.validate_config(config):
                return None
            
            # 处理路径
            config = self._process_paths(config)
            
            self.config = config
            self.logger.info(f"成功加载配置文件: {config_path}")
            return config
            
        except toml.TomlDecodeError as e:
            self.logger.error(f"TOML格式错误: {e}")
            return None
        except Exception as e:
            self.logger.error(f"加载配置文件时发生错误: {e}")
            return None
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置文件格式
        
        Args:
            config: 配置字典
            
        Returns:
            配置是否有效
        """
        required_sections = ['dataset', 'training', 'models']
        
        # 检查必需的配置节
        for section in required_sections:
            if section not in config:
                self.logger.error(f"配置文件缺少必需节: {section}")
                return False
        
        # 验证数据集配置
        if not self._validate_dataset_config(config['dataset']):
            return False
        
        # 验证训练配置
        if not self._validate_training_config(config['training']):
            return False
        
        # 验证模型配置
        if not self._validate_models_config(config['models']):
            return False
        
        return True
    
    def _validate_dataset_config(self, dataset_config: Dict[str, Any]) -> bool:
        """验证数据集配置"""
        required_fields = ['type', 'path', 'batch_size']
        
        for field in required_fields:
            if field not in dataset_config:
                self.logger.error(f"数据集配置缺少字段: {field}")
                return False
        
        # 验证数据集类型
        valid_types = ['small_binary', 'large_binary', 'small_multiclass', 'large_multiclass']
        if dataset_config['type'] not in valid_types:
            self.logger.error(f"无效的数据集类型: {dataset_config['type']}")
            return False
        
        # 验证批次大小
        if not isinstance(dataset_config['batch_size'], int) or dataset_config['batch_size'] <= 0:
            self.logger.error(f"无效的批次大小: {dataset_config['batch_size']}")
            return False
        
        return True
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> bool:
        """验证训练配置"""
        required_fields = ['epochs', 'learning_rate']
        
        for field in required_fields:
            if field not in training_config:
                self.logger.error(f"训练配置缺少字段: {field}")
                return False
        
        # 验证训练轮数
        if not isinstance(training_config['epochs'], int) or training_config['epochs'] <= 0:
            self.logger.error(f"无效的训练轮数: {training_config['epochs']}")
            return False
        
        # 验证学习率
        if not isinstance(training_config['learning_rate'], (int, float)) or training_config['learning_rate'] <= 0:
            self.logger.error(f"无效的学习率: {training_config['learning_rate']}")
            return False
        
        return True
    
    def _validate_models_config(self, models_config: Dict[str, Any]) -> bool:
        """验证模型配置"""
        if not models_config:
            self.logger.error("模型配置为空")
            return False
        
        # 检查是否有启用的模型
        enabled_models = [name for name, config in models_config.items() 
                         if config.get('enabled', True)]
        
        if not enabled_models:
            self.logger.error("没有启用的模型")
            return False
        
        # 验证每个模型配置
        for model_name, model_config in models_config.items():
            if not self._validate_single_model_config(model_name, model_config):
                return False
        
        return True
    
    def _validate_single_model_config(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """验证单个模型配置"""
        required_fields = ['module', 'class_name']
        
        for field in required_fields:
            if field not in model_config:
                self.logger.error(f"模型 {model_name} 配置缺少字段: {field}")
                return False
        
        return True
    
    def _process_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置中的路径，转换为绝对路径"""
        # 处理数据集路径
        if 'dataset' in config and 'path' in config['dataset']:
            dataset_path = config['dataset']['path']
            if not os.path.isabs(dataset_path):
                config['dataset']['path'] = os.path.abspath(dataset_path)
        
        # 处理输出路径
        if 'output' in config:
            for key in ['model_save_dir', 'results_save_dir', 'checkpoint_dir']:
                if key in config['output']:
                    path = config['output'][key]
                    if not os.path.isabs(path):
                        config['output'][key] = os.path.abspath(path)
        
        # 处理日志文件路径
        if 'logging' in config and 'file' in config['logging']:
            log_file = config['logging']['file']
            if not os.path.isabs(log_file):
                config['logging']['file'] = os.path.abspath(log_file)
        
        return config
    
    def get_dataset_config(self) -> Optional[Dict[str, Any]]:
        """获取数据集配置"""
        return self.config.get('dataset') if self.config else None
    
    def get_training_config(self) -> Optional[Dict[str, Any]]:
        """获取训练配置"""
        return self.config.get('training') if self.config else None
    
    def get_models_config(self) -> Optional[Dict[str, Any]]:
        """获取模型配置"""
        return self.config.get('models') if self.config else None
    
    def get_enabled_models(self) -> Dict[str, Dict[str, Any]]:
        """获取启用的模型配置"""
        if not self.config or 'models' not in self.config:
            return {}
        
        return {name: config for name, config in self.config['models'].items() 
                if config.get('enabled', True)}
    
    def get_output_config(self) -> Optional[Dict[str, Any]]:
        """获取输出配置"""
        return self.config.get('output') if self.config else None
    
    def get_device_config(self) -> Optional[Dict[str, Any]]:
        """获取设备配置"""
        return self.config.get('device') if self.config else None
    
    def get_experiment_config(self) -> Optional[Dict[str, Any]]:
        """获取实验配置"""
        return self.config.get('experiment') if self.config else None
    
    def get_logging_config(self) -> Optional[Dict[str, Any]]:
        """获取日志配置"""
        return self.config.get('logging') if self.config else None