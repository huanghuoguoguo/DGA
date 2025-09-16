#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 主入口脚本
统一的项目管理和执行入口
"""

import argparse
import sys
import os
import time
from datetime import datetime
import json
import torch
import numpy as np
from pathlib import Path


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, mode='max'):
        """
        Args:
            patience: 等待改善的轮数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
            mode: 'max' 表示指标越大越好(如准确率), 'min' 表示指标越小越好(如损失)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, score, model):
        """检查是否应该早停"""
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"早停触发，恢复最佳权重 (最佳分数: {self.best_score:.4f})")
            else:
                print(f"早停触发 (最佳分数: {self.best_score:.4f})")
        
        return self.early_stop

def main():
    """主函数 - 统一入口"""
    parser = argparse.ArgumentParser(
        description='DGA恶意域名检测 - 统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py test                    # 快速测试项目
  python main.py train --model cnn      # 训练CNN模型
  python main.py train --all --quick    # 快速训练所有模型
  python main.py analyze --chart        # 分析模型并生成图表
  python main.py train --help           # 查看训练选项
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='快速测试项目完整性')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--model', type=str, default='cnn',
                             choices=['cnn', 'lstm', 'mamba', 'moe', 'improved_moe', 'simplified_moe', 'mambaformer'],
                             help='要训练的模型类型')
    train_parser.add_argument('--quick', action='store_true',
                             help='快速测试模式（5个epoch）')
    train_parser.add_argument('--all', action='store_true',
                             help='训练所有模型')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析和对比模型')
    analyze_parser.add_argument('--chart', action='store_true',
                               help='生成对比图表')
    analyze_parser.add_argument('--expert', action='store_true',
                               help='分析专家使用情况')
    analyze_parser.add_argument('--output', type=str, 
                               default='./data/results/analysis_output.png',
                               help='图表输出路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应的命令
    if args.command == 'test':
        print("🧪 运行项目测试...")
        os.system('python quick_test.py')
        
    elif args.command == 'train':
        print("🚀 开始训练模型...")
        cmd = 'python simple_train.py'
        
        if args.all:
            cmd += ' --all'
        else:
            cmd += f' --model {args.model}'
            
        if args.quick:
            cmd += ' --quick'
            
        os.system(cmd)
        
    elif args.command == 'analyze':
        print("📊 开始分析模型...")
        cmd = 'python analyze_models.py'
        
        if args.chart:
            cmd += ' --chart'
        if args.expert:
            cmd += ' --expert'
        if args.output != './data/results/analysis_output.png':
            cmd += f' --output {args.output}'
            
        os.system(cmd)


if __name__ == "__main__":
    main()