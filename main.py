#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 主入口脚本
统一的项目管理和执行入口
"""

import argparse
import sys
import os

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
                             choices=['cnn', 'lstm', 'mamba', 'moe', 'mambaformer'],
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