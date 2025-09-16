# DGA恶意域名检测 - 配置驱动训练框架

## 🎯 项目简介

本项目实现了基于配置文件的热插拔DGA（Domain Generation Algorithm）恶意域名检测训练框架，支持多种深度学习架构，包括CNN、LSTM、TCBAM、MoE（混合专家）和Mamba模型。项目采用现代化的配置驱动架构，支持模型热插拔和统一训练流程。

## 🏗️ 项目结构

```
DGA/
├── core/                           # 核心模块
│   ├── dataset.py                  # 统一数据处理
│   ├── config_manager.py           # 配置管理器
│   ├── model_loader.py             # 动态模型加载器
│   ├── data_builder.py             # 数据集构建器
│   └── base_model.py               # 模型基类
├── models/                         # 模型定义
│   └── implementations/            # 具体模型实现
│       ├── cnn_model.py           # CNN模型
│       ├── simple_lstm_model.py   # LSTM模型
│       ├── tcbam_models.py        # TCBAM模型
│       ├── homogeneous_moe_model.py # MoE模型
│       └── mamba_model.py         # Mamba模型
├── config/                         # 配置管理
│   ├── config.py                  # 基础配置
│   └── train_config.toml          # 🔧 训练配置文件
├── data/                          # 数据目录
│   └── processed/                 # 预处理后的数据
│       ├── small_binary_dga_dataset.pkl
│       ├── large_binary_dga_dataset.pkl
│       ├── small_multiclass_dga_dataset.pkl
│       └── large_multiclass_dga_dataset.pkl
├── results/                       # 实验结果
│   ├── models/                    # 训练好的模型
│   └── experiments/               # 实验数据
├── train.py                       # 🚀 统一训练框架
├── main.py                        # 🎮 项目入口
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

### 2. 配置文件训练（推荐）

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置文件
python train.py --config ./config/train_config.toml

# 训练特定模型
python train.py --models lstm cnn
```

### 3. 通过统一入口训练

```bash
# 通过main.py训练
python main.py train --config ./config/train_config.toml
python main.py train --models lstm tcbam
```

### 4. 数据集构建

```bash
# 构建统一格式数据集
python core/data_builder.py
```

## 📊 支持的模型

| 模型 | 配置名 | 描述 | 优势 |
|------|--------|------|------|
| **CNN** | `cnn` | 多尺度卷积神经网络 | 速度快，轻量级，局部特征提取 |
| **LSTM** | `lstm` | 简单LSTM网络 | 序列建模，准确率高 |
| **TCBAM** | `tcbam` | 时间卷积+注意力机制 | 时序特征+注意力，性能优秀 |
| **MoE** | `moe` | 同构混合专家模型 | 专家协作，智能选择，优势互补 |
| **Mamba** | `mamba` | 状态空间模型 | 线性复杂度，长序列建模 |

## 🔧 配置文件说明

### 训练配置 (`config/train_config.toml`)

```toml
[dataset]
type = "small_binary"  # 数据集类型
path = "./data/processed/small_binary_dga_dataset.pkl"
batch_size = 32

[training]
epochs = 20
learning_rate = 0.001
weight_decay = 1e-4
optimizer = "adam"
scheduler = "plateau"

# 早停机制
early_stopping = true
patience = 5
min_delta = 0.001
monitor = "val_accuracy"
mode = "max"

[models.lstm]
module = "models.implementations.simple_lstm_model"
class_name = "SimpleLSTMModel"
enabled = true
params = { embed_dim = 128, hidden_dim = 128, num_layers = 2, dropout = 0.1 }

[models.cnn]
module = "models.implementations.cnn_model"
class_name = "CNNModel"
enabled = true
params = { embed_dim = 128, num_filters = 128, filter_sizes = [2, 3, 4], dropout = 0.1 }
```

## 💡 核心特性

### 🔧 配置驱动架构
- **TOML配置文件**: 所有训练参数通过配置文件管理
- **热插拔模型**: 通过配置文件动态加载模型，无需修改代码
- **模块化设计**: 配置管理、模型加载、训练流程完全分离
- **统一接口**: 所有模型使用相同的训练和评估接口

### 📦 智能模型加载
- **动态导入**: 根据配置文件动态导入模型类
- **参数验证**: 自动验证模型参数和配置格式
- **错误处理**: 完善的错误处理和用户友好的错误信息
- **缓存机制**: 模型类缓存，提高加载效率

### 🛡️ 鲁棒性设计
- **早停机制**: 可配置的早停策略，防止过拟合
- **学习率调度**: 支持多种学习率调度策略
- **梯度裁剪**: 防止梯度爆炸
- **设备自适应**: 自动检测和使用可用的计算设备

## 📈 数据集支持

### 统一格式数据集

| 数据集 | 样本数 | 类型 | 分割比例 |
|--------|--------|------|----------|
| `small_binary` | 10K | 二分类 | 8:1:1 |
| `large_binary` | 100K | 二分类 | 8:1:1 |
| `small_multiclass` | 10K | 16分类 | 8:1:1 |
| `large_multiclass` | 100K | 16分类 | 8:1:1 |

### 数据集特点
- **格式统一**: 所有数据集使用相同的格式和接口
- **标准分割**: 训练80%，验证10%，测试10%
- **分层抽样**: 确保各集合中类别分布一致
- **完整元信息**: 包含词汇表、类别分布、家族信息等

## 🔬 技术亮点

### 配置驱动训练
- **零代码修改**: 通过配置文件控制所有训练参数
- **模型热插拔**: 新增模型只需修改配置文件
- **实验管理**: 不同配置文件对应不同实验
- **可重现性**: 配置文件确保实验的完全可重现

### 统一训练框架
- **TrainingFramework**: 统一的训练流程管理
- **ConfigManager**: 配置文件加载和验证
- **ModelLoader**: 动态模型加载和实例化
- **EarlyStopping**: 可配置的早停机制

## 📝 使用示例

### 基础训练
```bash
# 使用默认配置训练所有启用的模型
python train.py

# 训练特定模型
python train.py --models lstm cnn

# 使用自定义配置
python train.py --config my_config.toml
```

### 配置文件定制
```toml
# 启用特定模型
[models.lstm]
enabled = true

[models.cnn]
enabled = false  # 禁用CNN模型

# 调整训练参数
[training]
epochs = 50
learning_rate = 0.0005
batch_size = 64
```

### 新增模型
```toml
# 在配置文件中添加新模型
[models.my_model]
module = "models.implementations.my_model"
class_name = "MyModel"
enabled = true
params = { param1 = 128, param2 = 0.1 }
```

## 📊 输出文件

训练完成后，会生成以下文件：

- **模型权重**: `results/models/{model_name}_best.pth`
- **实验结果**: `results/experiments/{experiment_name}_results.pkl`
- **训练日志**: `logs/training.log`
- **配置备份**: 自动保存使用的配置文件

## 🎯 项目优势

### 相比传统训练脚本
- ✅ **配置驱动**: 无需修改代码即可调整参数
- ✅ **模型解耦**: 训练框架与具体模型完全分离
- ✅ **热插拔**: 支持动态添加新模型
- ✅ **统一接口**: 所有模型使用相同的训练流程
- ✅ **实验管理**: 配置文件即实验记录

### 开发效率提升
- 🚀 **快速实验**: 修改配置文件即可开始新实验
- 🔧 **易于维护**: 清晰的模块分离，便于维护和扩展
- 📊 **结果对比**: 统一的结果格式，便于对比分析
- 🎮 **用户友好**: 简单的命令行接口，易于使用

## 🔧 扩展指南

### 添加新模型
1. 在 `models/implementations/` 中实现模型类
2. 在配置文件中添加模型配置
3. 运行训练即可，无需修改训练代码

### 自定义数据集
1. 实现数据集构建器
2. 生成统一格式的数据集文件
3. 在配置文件中指定数据集路径

### 调整训练策略
1. 修改配置文件中的训练参数
2. 支持不同的优化器、调度器、早停策略
3. 所有参数都可通过配置文件控制

## 📞 使用帮助

如果遇到问题：

1. 检查配置文件格式是否正确
2. 确认模型类路径和类名是否正确
3. 验证数据集文件是否存在
4. 查看日志文件获取详细错误信息

## 🎉 总结

重构后的项目具有：
- ✅ 配置驱动的现代化架构
- ✅ 模型热插拔能力
- ✅ 统一的训练和评估流程
- ✅ 完善的错误处理和日志记录
- ✅ 高度模块化和可扩展的设计

现在可以通过简单的配置文件管理复杂的深度学习实验！