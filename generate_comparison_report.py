#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能对比报告生成器
"""

import pickle
import json
import os
from datetime import datetime

def load_training_histories():
    """加载训练历史数据"""
    history_file = './data/training_logs/training_histories.pkl'
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            return pickle.load(f)
    return {}

def generate_detailed_report():
    """生成详细的性能对比报告"""
    
    # 从训练输出中提取的性能数据
    results = {
        'lstm': {
            'test_accuracy': 72.02,
            'f1_score': 0.6774,
            'inference_time_ms': 0.67,
            'best_val_accuracy': 72.06,
            'training_time_s': 149.40,
            'final_train_accuracy': 71.63,
            'final_val_accuracy': 72.02
        },
        'tcbam': {
            'test_accuracy': 71.87,
            'f1_score': 0.6754,
            'inference_time_ms': 7.76,
            'best_val_accuracy': 72.00,
            'training_time_s': 375.05,
            'final_train_accuracy': 71.62,
            'final_val_accuracy': 71.98
        }
    }
    
    # 生成报告
    report = []
    report.append("="*80)
    report.append("超大规模多分类DGA检测模型性能对比报告")
    report.append("="*80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据集: 超大规模多分类DGA数据集")
    report.append(f"训练轮数: 15 epochs")
    report.append("")
    
    # 模型概述
    report.append("模型概述:")
    report.append("-" * 40)
    report.append("• LSTM: 长短期记忆网络，专门处理序列数据")
    report.append("• TCBAM: 时间卷积块注意力机制模型，结合了时间卷积和注意力机制")
    report.append("")
    
    # 详细性能对比表格
    report.append("详细性能对比:")
    report.append("-" * 80)
    report.append(f"{'指标':<20} {'LSTM':<15} {'TCBAM':<15} {'最佳模型':<15}")
    report.append("-" * 80)
    
    # 准确率对比
    lstm_acc = results['lstm']['test_accuracy']
    tcbam_acc = results['tcbam']['test_accuracy']
    best_acc_model = 'LSTM' if lstm_acc > tcbam_acc else 'TCBAM'
    report.append(f"{'测试准确率 (%)':<20} {lstm_acc:<15.2f} {tcbam_acc:<15.2f} {best_acc_model:<15}")
    
    # F1分数对比
    lstm_f1 = results['lstm']['f1_score']
    tcbam_f1 = results['tcbam']['f1_score']
    best_f1_model = 'LSTM' if lstm_f1 > tcbam_f1 else 'TCBAM'
    report.append(f"{'F1分数':<20} {lstm_f1:<15.4f} {tcbam_f1:<15.4f} {best_f1_model:<15}")
    
    # 推理时间对比
    lstm_time = results['lstm']['inference_time_ms']
    tcbam_time = results['tcbam']['inference_time_ms']
    best_time_model = 'LSTM' if lstm_time < tcbam_time else 'TCBAM'
    report.append(f"{'推理时间 (ms)':<20} {lstm_time:<15.2f} {tcbam_time:<15.2f} {best_time_model:<15}")
    
    # 训练时间对比
    lstm_train_time = results['lstm']['training_time_s']
    tcbam_train_time = results['tcbam']['training_time_s']
    best_train_time_model = 'LSTM' if lstm_train_time < tcbam_train_time else 'TCBAM'
    report.append(f"{'训练时间 (s)':<20} {lstm_train_time:<15.2f} {tcbam_train_time:<15.2f} {best_train_time_model:<15}")
    
    # 最佳验证准确率
    lstm_best_val = results['lstm']['best_val_accuracy']
    tcbam_best_val = results['tcbam']['best_val_accuracy']
    best_val_model = 'LSTM' if lstm_best_val > tcbam_best_val else 'TCBAM'
    report.append(f"{'最佳验证准确率 (%)':<20} {lstm_best_val:<15.2f} {tcbam_best_val:<15.2f} {best_val_model:<15}")
    
    report.append("")
    
    # 性能分析
    report.append("性能分析:")
    report.append("-" * 40)
    
    # 准确率分析
    acc_diff = abs(lstm_acc - tcbam_acc)
    report.append(f"• 准确率差异: {acc_diff:.2f}% (LSTM vs TCBAM)")
    if acc_diff < 0.5:
        report.append("  两个模型在准确率上表现相当接近")
    elif lstm_acc > tcbam_acc:
        report.append("  LSTM在准确率上略胜一筹")
    else:
        report.append("  TCBAM在准确率上略胜一筹")
    
    # 效率分析
    speed_ratio = tcbam_time / lstm_time
    report.append(f"• 推理速度: LSTM比TCBAM快 {speed_ratio:.1f}x")
    
    # 训练效率分析
    train_speed_ratio = tcbam_train_time / lstm_train_time
    report.append(f"• 训练效率: LSTM比TCBAM快 {train_speed_ratio:.1f}x")
    
    report.append("")
    
    # 综合评估
    report.append("综合评估:")
    report.append("-" * 40)
    
    # 计算综合得分 (准确率权重0.4, F1权重0.3, 速度权重0.3)
    lstm_score = (lstm_acc * 0.4 + lstm_f1 * 100 * 0.3 + (1/lstm_time) * 10 * 0.3)
    tcbam_score = (tcbam_acc * 0.4 + tcbam_f1 * 100 * 0.3 + (1/tcbam_time) * 10 * 0.3)
    
    report.append(f"• LSTM综合得分: {lstm_score:.2f}")
    report.append(f"• TCBAM综合得分: {tcbam_score:.2f}")
    
    if lstm_score > tcbam_score:
        report.append("• 推荐模型: LSTM")
        report.append("  理由: 在保持相近准确率的同时，具有显著的速度优势")
    else:
        report.append("• 推荐模型: TCBAM")
        report.append("  理由: 在准确率和F1分数上表现更好")
    
    report.append("")
    
    # 应用建议
    report.append("应用建议:")
    report.append("-" * 40)
    report.append("• 实时检测场景: 推荐使用LSTM，因为其推理速度更快")
    report.append("• 批量处理场景: 两个模型都适用，可根据准确率需求选择")
    report.append("• 资源受限环境: 推荐使用LSTM，训练和推理都更高效")
    report.append("• 高精度要求: 两个模型准确率相近，可根据其他因素选择")
    
    report.append("")
    report.append("="*80)
    
    return '\n'.join(report)

def save_report(report_content):
    """保存报告到文件"""
    os.makedirs('./data/reports', exist_ok=True)
    
    # 保存文本报告
    with open('./data/reports/model_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 保存JSON格式的数据
    results_json = {
        'lstm': {
            'test_accuracy': 72.02,
            'f1_score': 0.6774,
            'inference_time_ms': 0.67,
            'best_val_accuracy': 72.06,
            'training_time_s': 149.40
        },
        'tcbam': {
            'test_accuracy': 71.87,
            'f1_score': 0.6754,
            'inference_time_ms': 7.76,
            'best_val_accuracy': 72.00,
            'training_time_s': 375.05
        },
        'comparison': {
            'accuracy_difference': 0.15,
            'speed_ratio': 11.6,
            'recommended_model': 'LSTM',
            'recommendation_reason': '在保持相近准确率的同时，具有显著的速度优势'
        }
    }
    
    with open('./data/reports/model_comparison_data.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    print("生成模型性能对比报告...")
    
    # 生成报告
    report = generate_detailed_report()
    
    # 保存报告
    save_report(report)
    
    # 打印报告
    print(report)
    
    print("\n报告已保存到:")
    print("- ./data/reports/model_comparison_report.txt")
    print("- ./data/reports/model_comparison_data.json")