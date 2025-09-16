#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态模型加载器
支持基于配置文件的模型热插拔
"""

import importlib
import logging
from typing import Dict, Any, Type, Optional
import torch.nn as nn


class ModelLoader:
    """动态模型加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._model_cache = {}  # 模型类缓存
    
    def load_model_class(self, module_path: str, class_name: str) -> Optional[Type[nn.Module]]:
        """动态加载模型类
        
        Args:
            module_path: 模块路径，如 'models.implementations.lstm_model'
            class_name: 类名，如 'LSTMModel'
            
        Returns:
            模型类或None（如果加载失败）
        """
        cache_key = f"{module_path}.{class_name}"
        
        # 检查缓存
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            # 获取模型类
            if not hasattr(module, class_name):
                self.logger.error(f"模块 {module_path} 中未找到类 {class_name}")
                return None
            
            model_class = getattr(module, class_name)
            
            # 验证是否为nn.Module子类
            if not issubclass(model_class, nn.Module):
                self.logger.error(f"类 {class_name} 不是 nn.Module 的子类")
                return None
            
            # 缓存模型类
            self._model_cache[cache_key] = model_class
            self.logger.info(f"成功加载模型类: {module_path}.{class_name}")
            
            return model_class
            
        except ImportError as e:
            self.logger.error(f"导入模块失败 {module_path}: {e}")
            return None
        except AttributeError as e:
            self.logger.error(f"获取类失败 {class_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"加载模型类时发生未知错误: {e}")
            return None
    
    def create_model(self, model_config: Dict[str, Any], dataset_info: Dict[str, Any]) -> Optional[nn.Module]:
        """创建模型实例
        
        Args:
            model_config: 模型配置，包含module, class_name, params等
            dataset_info: 数据集信息，包含vocab_size, num_classes等
            
        Returns:
            模型实例或None（如果创建失败）
        """
        try:
            # 加载模型类
            model_class = self.load_model_class(
                model_config['module'], 
                model_config['class_name']
            )
            
            if model_class is None:
                return None
            
            # 准备模型参数
            model_params = model_config.get('params', {}).copy()
            
            # 添加数据集相关参数
            model_params['vocab_size'] = dataset_info['vocab_size']
            model_params['num_classes'] = dataset_info['num_classes']
            
            # 创建模型实例
            model = model_class(**model_params)
            
            self.logger.info(f"成功创建模型: {model_config['class_name']}")
            return model
            
        except TypeError as e:
            self.logger.error(f"模型参数错误: {e}")
            return None
        except Exception as e:
            self.logger.error(f"创建模型时发生错误: {e}")
            return None
    
    def get_available_models(self, models_config: Dict[str, Dict[str, Any]]) -> Dict[str, Type[nn.Module]]:
        """获取所有可用的模型类
        
        Args:
            models_config: 模型配置字典
            
        Returns:
            可用模型类字典
        """
        available_models = {}
        
        for model_name, config in models_config.items():
            if not config.get('enabled', True):
                continue
                
            model_class = self.load_model_class(
                config['module'], 
                config['class_name']
            )
            
            if model_class is not None:
                available_models[model_name] = model_class
        
        self.logger.info(f"找到 {len(available_models)} 个可用模型: {list(available_models.keys())}")
        return available_models
    
    def validate_model_config(self, model_config: Dict[str, Any]) -> bool:
        """验证模型配置
        
        Args:
            model_config: 模型配置
            
        Returns:
            配置是否有效
        """
        required_fields = ['module', 'class_name']
        
        for field in required_fields:
            if field not in model_config:
                self.logger.error(f"模型配置缺少必需字段: {field}")
                return False
        
        # 检查模块路径格式
        module_path = model_config['module']
        if not isinstance(module_path, str) or not module_path:
            self.logger.error(f"无效的模块路径: {module_path}")
            return False
        
        # 检查类名格式
        class_name = model_config['class_name']
        if not isinstance(class_name, str) or not class_name:
            self.logger.error(f"无效的类名: {class_name}")
            return False
        
        return True
    
    def clear_cache(self):
        """清空模型类缓存"""
        self._model_cache.clear()
        self.logger.info("模型类缓存已清空")