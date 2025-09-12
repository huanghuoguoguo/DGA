# DGA恶意域名检测 - 重构版

## 🎯 项目简介

本项目实现了多种深度学习架构用于DGA（Domain Generation Algorithm）恶意域名检测，包括CNN、LSTM、Mamba和MoE（混合专家）模型。项目已经过重构，具有清晰的模块化结构和简化的使用流程。

## 🏗️ 项目结构

```
DGA/
├── core/                           # 核心模块
│   ├── __init__.py
│   ├── dataset.py                  # 统一数据处理
│   └── base_model.py               # 模型基类和训练器
├── models/                         # 模型定义
│   ├── base/                       # 基础模型组件
│   └── implementations/            # 具体模型实现
│       ├── cnn_model.py           # CNN模型
│       ├── lstm_model.py          # BiLSTM+Attention模型
│       ├── mamba_model.py         # Mamba模型（状态空间模型）
│       └── moe_model.py           # 混合专家模型
├── config/                         # 配置管理
│   └── config.py                  # 统一配置
├── data/                          # 数据目录
│   ├── processed/                 # 预处理后的数据
│   ├── models/                    # 训练好的模型
│   ├── results/                   # 实验结果
│   └── raw/                       # 原始数据
├── scripts/                       # 脚本目录
│   ├── train/                     # 训练脚本
│   ├── eval/                      # 评估脚本
│   └── analysis/                  # 分析脚本
├── simple_train.py                # 🚀 简化训练脚本
├── quick_test.py                  # 🧪 快速测试脚本
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 1. 环境检查

```bash
# 激活conda环境
conda activate DGAenv

# 检查PyTorch安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

### 2. 项目验证

```bash
# 运行快速测试，验证所有组件
python quick_test.py
```

### 3. 快速训练测试

```bash
# 快速训练CNN模型（5个epoch）
python simple_train.py --model cnn --quick

# 快速训练Mamba模型
python simple_train.py --model mamba --quick
```

### 4. 完整训练

```bash
# 训练单个模型
python simple_train.py --model cnn
python simple_train.py --model lstm
python simple_train.py --model mamba
python simple_train.py --model moe

# 训练所有模型
python simple_train.py --all
```

## 📊 支持的模型

| 模型 | 描述 | 优势 |
|------|------|------|
| **CNN** | 多尺度卷积神经网络 | 速度快，轻量级，局部特征提取 |
| **LSTM** | BiLSTM+注意力机制 | 序列建模强，准确率高 |
| **Mamba** | 状态空间模型 | 线性复杂度，长序列建模，选择性机制 |
| **MoE** | 混合专家模型 | 专家协作，智能选择，优势互补 |

## 💡 核心特性

### 🔧 模块化设计
- **统一的基类**: 所有模型继承`BaseModel`，接口一致
- **统一的训练器**: `ModelTrainer`处理所有模型的训练流程
- **统一的数据处理**: `core.dataset`模块处理数据加载和预处理
- **统一的配置管理**: `config.config`管理所有超参数

### 📦 简化的使用流程
- **一键训练**: `simple_train.py`脚本支持训练任意模型
- **快速测试**: `quick_test.py`验证项目完整性
- **自动保存**: 模型和结果自动保存到指定位置
- **进度显示**: 训练过程带有进度条和详细信息

### 🛡️ 鲁棒性设计
- **错误处理**: 完善的异常处理和用户友好的错误信息
- **路径管理**: 自动创建必要目录，支持多种数据集路径
- **设备自适应**: 自动检测和使用可用的计算设备

## 📈 性能表现

基于之前的实验结果：

| 模型 | 准确率 | F1分数 | 推理时间 | 参数量 |
|------|--------|--------|----------|--------|
| CNN | 94.44% | 0.944 | 2.66ms | 195K |
| BiLSTM+Attention | 95.28% | 0.953 | 9.55ms | 689K |
| Mamba (预测) | ~94.5% | ~0.945 | ~8.5ms | ~450K |
| MoE | 94.44% | 0.944 | 9.17ms | 888K |

## 🔬 技术亮点

### Mamba模型
- **线性复杂度**: O(L) vs Transformer的O(L²)
- **选择性机制**: 动态选择重要信息
- **状态空间模型**: 高效的长序列建模

### MoE架构
- **专家混合**: 结合不同模型的优势
- **门控网络**: 智能选择最适合的专家
- **负载均衡**: 确保专家的合理使用

## 📝 使用示例

```python
# 训练Mamba模型
python simple_train.py --model mamba

# 快速测试所有模型
python simple_train.py --all --quick

# 项目完整性检查
python quick_test.py
```

## 🔧 配置说明

主要配置在 `config/config.py` 中：

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 5
    random_seed: int = 42
```

## 📊 输出文件

训练完成后，会生成以下文件：

- **模型权重**: `data/models/best_{model_name}_model.pth`
- **实验结果**: `data/results/{model_name}_results.pkl`
- **训练日志**: 控制台输出包含详细的训练信息

## 🎯 下一步计划

1. **模型优化**: 超参数调优和架构改进
2. **数据扩展**: 增加更多DGA家族数据
3. **性能分析**: 详细的模型对比和分析
4. **部署优化**: 模型压缩和推理加速

## 📞 使用帮助

如果遇到问题：

1. 首先运行 `python quick_test.py` 检查环境
2. 确认数据集文件 `data/processed/small_dga_dataset.pkl` 存在
3. 检查PyTorch是否正确安装
4. 使用 `--quick` 选项进行快速测试

## 🎉 总结

重构后的项目具有：
- ✅ 清晰的模块化结构
- ✅ 简化的使用流程  
- ✅ 统一的接口设计
- ✅ 完善的错误处理
- ✅ 详细的文档说明

现在可以轻松地训练和对比不同的DGA检测模型！