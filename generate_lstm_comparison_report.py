#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM模型对比报告生成器
对比简单LSTM与BiLSTM+Attention模型的性能差异
"""

import json
import os
from datetime import datetime

def generate_lstm_comparison_report():
    """
    生成LSTM模型对比报告
    """
    
    # 模型性能数据（基于训练结果）
    model_results = {
        'simple_lstm': {
            'accuracy': 70.88,
            'f1_score': 0.6683,
            'avg_loss': 0.6801,
            'inference_time_ms': 0.62,
            'model_complexity': '简单',
            'architecture': '单向LSTM + 全连接层',
            'parameters': '相对较少',
            'training_time': '较快'
        },
        'bilstm_attention': {
            'accuracy': 71.11,
            'f1_score': 0.6765,
            'avg_loss': 0.6583,
            'inference_time_ms': 0.65,
            'model_complexity': '复杂',
            'architecture': '双向LSTM + 注意力机制 + 全连接层',
            'parameters': '相对较多',
            'training_time': '较慢'
        }
    }
    
    # 计算性能差异
    accuracy_diff = model_results['bilstm_attention']['accuracy'] - model_results['simple_lstm']['accuracy']
    f1_diff = model_results['bilstm_attention']['f1_score'] - model_results['simple_lstm']['f1_score']
    loss_diff = model_results['simple_lstm']['avg_loss'] - model_results['bilstm_attention']['avg_loss']
    time_diff = model_results['bilstm_attention']['inference_time_ms'] - model_results['simple_lstm']['inference_time_ms']
    
    # 生成详细报告
    report = f"""
================================================================================
LSTM模型性能对比分析报告
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据集: xlarge_multiclass_dga_dataset.pkl
训练轮数: 5 epochs

1. 模型架构对比
================================================================================

简单LSTM模型 (Simple LSTM):
- 架构: {model_results['simple_lstm']['architecture']}
- 复杂度: {model_results['simple_lstm']['model_complexity']}
- 参数量: {model_results['simple_lstm']['parameters']}
- 训练速度: {model_results['simple_lstm']['training_time']}

BiLSTM+Attention模型:
- 架构: {model_results['bilstm_attention']['architecture']}
- 复杂度: {model_results['bilstm_attention']['model_complexity']}
- 参数量: {model_results['bilstm_attention']['parameters']}
- 训练速度: {model_results['bilstm_attention']['training_time']}

2. 性能指标对比
================================================================================

{'模型名称':<20} {'准确率':<12} {'F1分数':<12} {'平均损失':<12} {'推理时间(ms)':<15}
{'-'*75}
{'Simple LSTM':<20} {model_results['simple_lstm']['accuracy']:<12.2f} {model_results['simple_lstm']['f1_score']:<12.4f} {model_results['simple_lstm']['avg_loss']:<12.4f} {model_results['simple_lstm']['inference_time_ms']:<15.2f}
{'BiLSTM+Attention':<20} {model_results['bilstm_attention']['accuracy']:<12.2f} {model_results['bilstm_attention']['f1_score']:<12.4f} {model_results['bilstm_attention']['avg_loss']:<12.4f} {model_results['bilstm_attention']['inference_time_ms']:<15.2f}

3. 性能差异分析
================================================================================

准确率差异: {accuracy_diff:+.2f}% (BiLSTM+Attention {'优于' if accuracy_diff > 0 else '劣于'} Simple LSTM)
F1分数差异: {f1_diff:+.4f} (BiLSTM+Attention {'优于' if f1_diff > 0 else '劣于'} Simple LSTM)
损失差异: {loss_diff:+.4f} (BiLSTM+Attention损失{'更低' if loss_diff > 0 else '更高'})
推理时间差异: {time_diff:+.2f}ms (BiLSTM+Attention {'更慢' if time_diff > 0 else '更快'})

4. 详细分析
================================================================================

4.1 准确率分析:
- BiLSTM+Attention模型在准确率上略优于Simple LSTM ({accuracy_diff:.2f}%的提升)
- 双向LSTM能够捕获序列的前后文信息，注意力机制进一步增强了重要特征的权重
- 提升幅度相对较小，说明对于DGA检测任务，简单模型已能获得不错效果

4.2 F1分数分析:
- BiLSTM+Attention在F1分数上也有小幅提升 ({f1_diff:.4f})
- 表明复杂模型在精确率和召回率的平衡上稍有优势
- 但提升幅度有限，性价比需要考虑

4.3 损失函数分析:
- BiLSTM+Attention的平均损失更低 ({abs(loss_diff):.4f})
- 说明复杂模型在训练数据上的拟合效果更好
- 但需要注意是否存在过拟合风险

4.4 推理时间分析:
- BiLSTM+Attention的推理时间略长 ({time_diff:.2f}ms)
- 增加的计算复杂度带来了轻微的性能开销
- 对于实时应用场景需要权衡准确率提升与延迟增加

5. 模型选择建议
================================================================================

5.1 选择Simple LSTM的场景:
- 对推理速度要求较高的实时检测系统
- 计算资源受限的边缘设备部署
- 追求模型简洁性和可解释性的场景
- 对准确率要求不是极致苛刻的应用

5.2 选择BiLSTM+Attention的场景:
- 对检测准确率有更高要求的安全关键应用
- 计算资源充足的服务器端部署
- 需要更好的序列建模能力的复杂DGA检测
- 可以接受轻微性能开销换取准确率提升的场景

6. 总结
================================================================================

本次对比实验表明:

1. **性能差异**: BiLSTM+Attention模型在各项指标上均略优于Simple LSTM，但提升幅度有限

2. **复杂度权衡**: 复杂模型带来的性能提升需要与增加的计算开销进行权衡

3. **实用性考虑**: 对于DGA检测任务，Simple LSTM已能提供相当不错的性能，是一个高性价比的选择

4. **应用建议**: 根据具体应用场景的需求（准确率vs速度vs资源消耗）来选择合适的模型

5. **进一步优化**: 可以考虑模型压缩、知识蒸馏等技术来平衡性能和效率

================================================================================
报告生成完成
================================================================================
"""
    
    # 保存文本报告
    os.makedirs('./data/reports', exist_ok=True)
    report_file = './data/reports/lstm_models_comparison_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存JSON格式的数据
    json_data = {
        'report_info': {
            'title': 'LSTM模型性能对比分析报告',
            'generated_time': datetime.now().isoformat(),
            'dataset': 'xlarge_multiclass_dga_dataset.pkl',
            'epochs': 5
        },
        'model_results': model_results,
        'performance_differences': {
            'accuracy_diff': accuracy_diff,
            'f1_diff': f1_diff,
            'loss_diff': loss_diff,
            'inference_time_diff': time_diff
        },
        'recommendations': {
            'simple_lstm_scenarios': [
                '对推理速度要求较高的实时检测系统',
                '计算资源受限的边缘设备部署',
                '追求模型简洁性和可解释性的场景',
                '对准确率要求不是极致苛刻的应用'
            ],
            'bilstm_attention_scenarios': [
                '对检测准确率有更高要求的安全关键应用',
                '计算资源充足的服务器端部署',
                '需要更好的序列建模能力的复杂DGA检测',
                '可以接受轻微性能开销换取准确率提升的场景'
            ]
        }
    }
    
    json_file = './data/reports/lstm_models_comparison_data.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print("LSTM模型对比报告生成完成!")
    print(f"文本报告: {report_file}")
    print(f"JSON数据: {json_file}")
    print("\n" + "="*80)
    print("主要发现:")
    print(f"- BiLSTM+Attention准确率: {model_results['bilstm_attention']['accuracy']:.2f}%")
    print(f"- Simple LSTM准确率: {model_results['simple_lstm']['accuracy']:.2f}%")
    print(f"- 准确率提升: {accuracy_diff:+.2f}%")
    print(f"- F1分数提升: {f1_diff:+.4f}")
    print(f"- 推理时间增加: {time_diff:+.2f}ms")
    print("="*80)
    
    return report_file, json_file

if __name__ == '__main__':
    generate_lstm_comparison_report()