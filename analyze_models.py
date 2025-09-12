#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGA Detection - 统一模型分析和对比脚本
"""

import torch
import pickle
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from core.dataset import create_data_loaders, print_dataset_info
    from core.base_model import ModelTrainer
    from config.config import config
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保项目结构完整")


def load_all_results():
    """加载所有模型的实验结果"""
    results = {}
    results_dir = config.paths['results_dir']
    
    # 查找所有结果文件
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('_results.pkl'):
                model_name = file.replace('_results.pkl', '').replace('_', ' ').title()
                try:
                    with open(os.path.join(results_dir, file), 'rb') as f:
                        results[model_name] = pickle.load(f)
                    print(f"✅ 加载结果: {model_name}")
                except Exception as e:
                    print(f"❌ 加载失败 {file}: {e}")
    
    # 尝试加载legacy结果文件
    legacy_files = [
        ('./data/training_results.pkl', 'CNN'),
        ('./data/final_test_results.pkl', 'Final Test'),
    ]
    
    for file_path, model_type in legacy_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if model_type == 'Final Test' and isinstance(data, dict):
                        # 解析final_test_results.pkl中的多个模型
                        for name, result in data.items():
                            if isinstance(result, dict) and 'accuracy' in result:
                                results[name] = {
                                    'test_results': result,
                                    'model_info': {'total_params': result.get('params', 0)}
                                }
                    else:
                        results[model_type] = data
                print(f"✅ 加载legacy结果: {model_type}")
            except Exception as e:
                print(f"❌ 加载失败 {file_path}: {e}")
    
    return results


def print_comparison_table(results):
    """打印模型对比表格"""
    if not results:
        print("❌ 没有找到任何模型结果")
        return
    
    print("\n" + "="*100)
    print("🔍 DGA恶意域名检测 - 模型性能对比")
    print("="*100)
    
    # 表头
    print(f"{'模型':<25} {'准确率':<10} {'F1分数':<10} {'推理时间(ms)':<12} {'参数量':<12} {'大小(MB)':<10}")
    print("-" * 100)
    
    # 整理数据
    model_data = []
    for model_name, data in results.items():
        try:
            # 提取测试结果
            if 'test_results' in data:
                test_data = data['test_results']
                accuracy = test_data.get('accuracy', 0)
                f1_score = test_data.get('f1', 0)
                inference_time = test_data.get('inference_time_ms', 0)
            else:
                # 兼容旧格式
                accuracy = data.get('test_accuracy', data.get('accuracy', 0))
                f1_score = data.get('test_f1', data.get('f1', 0))
                inference_time = data.get('avg_inference_time_ms', data.get('inference_time_ms', 0))
            
            # 提取模型信息
            if 'model_info' in data:
                model_info = data['model_info']
                params = model_info.get('total_params', 0)
            else:
                params = data.get('model_params', data.get('params', 0))
            
            # 格式化数据
            if accuracy < 1:
                accuracy *= 100
            
            size_mb = params * 4 / 1024 / 1024 if params > 0 else 0
            
            model_data.append({
                'name': model_name,
                'accuracy': accuracy,
                'f1': f1_score,
                'inference_time': inference_time,
                'params': params,
                'size_mb': size_mb
            })
            
        except Exception as e:
            print(f"⚠️  解析模型数据失败 {model_name}: {e}")
            continue
    
    # 按准确率排序
    model_data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # 打印数据
    for data in model_data:
        print(f"{data['name']:<25} {data['accuracy']:<9.2f}% {data['f1']:<9.4f} "
              f"{data['inference_time']:<11.2f} {data['params']:<11,} {data['size_mb']:<9.2f}")
    
    print("="*100)
    
    # 性能总结
    if model_data:
        best_acc = max(model_data, key=lambda x: x['accuracy'])
        best_f1 = max(model_data, key=lambda x: x['f1'])
        fastest = min(model_data, key=lambda x: x['inference_time'] if x['inference_time'] > 0 else float('inf'))
        smallest = min(model_data, key=lambda x: x['params'] if x['params'] > 0 else float('inf'))
        
        print(f"\n🏆 性能总结:")
        print(f"  最高准确率: {best_acc['name']} ({best_acc['accuracy']:.2f}%)")
        print(f"  最佳F1分数: {best_f1['name']} ({best_f1['f1']:.4f})")
        print(f"  最快推理: {fastest['name']} ({fastest['inference_time']:.2f}ms)")
        print(f"  最小模型: {smallest['name']} ({smallest['params']:,}参数)")
    
    return model_data


def create_comparison_chart(model_data, output_path='./data/results/model_comparison.png'):
    """创建模型对比图表"""
    if not model_data:
        return
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    names = [d['name'] for d in model_data]
    accuracies = [d['accuracy'] for d in model_data]
    f1_scores = [d['f1'] for d in model_data]
    inference_times = [d['inference_time'] for d in model_data if d['inference_time'] > 0]
    inf_names = [d['name'] for d in model_data if d['inference_time'] > 0]
    params = [d['params']/1000 for d in model_data if d['params'] > 0]  # 转换为K
    param_names = [d['name'] for d in model_data if d['params'] > 0]
    
    # 准确率对比
    bars1 = ax1.bar(names, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    # F1分数对比
    bars2 = ax2.bar(names, f1_scores, color='lightgreen', alpha=0.7)
    ax2.set_title('Model F1-Score Comparison')
    ax2.set_ylabel('F1-Score')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 推理时间对比
    if inference_times:
        bars3 = ax3.bar(inf_names, inference_times, color='salmon', alpha=0.7)
        ax3.set_title('Model Inference Time Comparison')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(inference_times):
            ax3.text(i, v + max(inference_times)*0.02, f'{v:.1f}', ha='center', va='bottom')
    
    # 参数量对比
    if params:
        bars4 = ax4.bar(param_names, params, color='gold', alpha=0.7)
        ax4.set_title('Model Parameters Comparison')
        ax4.set_ylabel('Parameters (K)')
        ax4.tick_params(axis='x', rotation=45)
        for i, v in enumerate(params):
            ax4.text(i, v + max(params)*0.02, f'{v:.0f}K', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 对比图表已保存到: {output_path}")


def analyze_expert_usage(results):
    """分析MoE模型的专家使用情况"""
    print(f"\n🔧 MoE模型专家使用分析:")
    
    for model_name, data in results.items():
        if 'moe' in model_name.lower() or 'expert' in model_name.lower():
            if 'test_results' in data and 'expert_usage' in data['test_results']:
                usage = data['test_results']['expert_usage']
                print(f"  {model_name}:")
                if isinstance(usage, dict):
                    for expert, ratio in usage.items():
                        print(f"    {expert}: {ratio:.1%}")
                elif isinstance(usage, (list, tuple)):
                    expert_names = ['CNN', 'LSTM', 'Transformer', 'Mamba']
                    for i, ratio in enumerate(usage):
                        if i < len(expert_names):
                            print(f"    {expert_names[i]}: {ratio:.1%}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DGA检测模型分析和对比')
    parser.add_argument('--chart', action='store_true', help='生成对比图表')
    parser.add_argument('--expert', action='store_true', help='分析专家使用情况')
    parser.add_argument('--output', type=str, default='./data/results/analysis_output.png',
                       help='图表输出路径')
    
    args = parser.parse_args()
    
    print("🚀 DGA检测模型分析工具")
    print("=" * 50)
    
    # 加载所有结果
    print("📂 加载模型结果...")
    results = load_all_results()
    
    if not results:
        print("❌ 没有找到任何模型结果文件")
        print("请先运行训练脚本: python simple_train.py --all")
        return
    
    # 打印对比表格
    model_data = print_comparison_table(results)
    
    # 分析专家使用情况
    if args.expert:
        analyze_expert_usage(results)
    
    # 生成对比图表
    if args.chart and model_data:
        create_comparison_chart(model_data, args.output)
    
    print(f"\n✅ 分析完成！共分析了 {len(results)} 个模型")


if __name__ == "__main__":
    main()