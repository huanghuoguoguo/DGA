# DGA恶意域名检测项目 - 开发与调试规则

## 📋 项目概述

本项目是一个基于深度学习的DGA（Domain Generation Algorithm）恶意域名检测系统，采用模块化架构，支持多种模型（CNN、LSTM、Mamba、MoE、MambaFormer）的训练、评估和对比。

---

## 🏗️ 代码架构规则

### 1. 模块化设计原则

#### 1.1 核心模块分离
- **`core/`**: 包含基础组件，所有模型共享
  - `base_model.py`: 所有模型必须继承`BaseModel`类
  - `dataset.py`: 统一数据处理接口
- **`models/implementations/`**: 具体模型实现
- **`config/`**: 统一配置管理

#### 1.2 接口一致性
```python
# ✅ 正确：所有模型继承BaseModel
class NewModel(BaseModel):
    def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1):
        super().__init__(vocab_size, num_classes)
    
    def forward(self, x):
        # 实现前向传播
        pass

# ❌ 错误：直接继承nn.Module
class BadModel(nn.Module):
    pass
```

#### 1.3 路径管理规范
```python
# ✅ 正确：使用配置管理路径
from config.config import config
save_path = config.get_model_save_path(model_name)

# ❌ 错误：硬编码路径
save_path = './data/models/model.pth'
```

---

## 🔧 模型开发规则

### 2. 模型实现标准

#### 2.1 必需方法
每个模型必须实现：
- `__init__()`: 初始化模型结构
- `forward()`: 前向传播
- `get_model_info()`: 继承自BaseModel，无需重写

#### 2.2 参数规范
```python
# ✅ 标准参数接口
def __init__(self, vocab_size, d_model=128, num_classes=2, dropout=0.1, **kwargs):
    # vocab_size: 必需参数
    # d_model: 模型维度，默认128
    # num_classes: 分类数，默认2（二分类）
    # dropout: dropout率，默认0.1
    # **kwargs: 模型特定参数
```

#### 2.3 权重初始化
```python
# ✅ 推荐的权重初始化
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
```

### 3. 模型性能要求

#### 3.1 基准性能
- **准确率**: 新模型应达到 ≥ 93%
- **推理时间**: 单样本推理 ≤ 20ms
- **参数量**: 尽量控制在 ≤ 1M 参数

#### 3.2 内存管理
```python
# ✅ 正确：使用torch.no_grad()进行推理
with torch.no_grad():
    output = model(data)

# ✅ 正确：及时清理梯度
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 📊 数据处理规则

### 4. 数据集管理

#### 4.1 数据路径优先级
1. `./data/processed/small_dga_dataset.pkl`
2. `./data/small_dga_dataset.pkl`
3. 其他自定义路径

#### 4.2 数据加载规范
```python
# ✅ 正确：使用统一的数据加载接口
from core.dataset import create_data_loaders
train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
    dataset_path=config.data.dataset_path,
    batch_size=config.training.batch_size
)

# ❌ 错误：自定义数据加载
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
```

#### 4.3 数据预处理
- 序列长度统一为40
- 使用padding_idx=0进行填充
- 词汇表大小根据数据集自动确定

---

## 🚀 训练与实验规则

### 5. 训练流程标准

#### 5.1 训练脚本使用
```bash
# ✅ 推荐：使用统一训练脚本
python simple_train.py --model cnn --quick  # 快速测试
python simple_train.py --model mamba        # 完整训练
python simple_train.py --all               # 训练所有模型

# ✅ 推荐：使用主入口
python main.py train --model cnn --quick
python main.py test                         # 项目测试
python main.py analyze --chart             # 结果分析
```

#### 5.2 超参数配置
```python
# ✅ 正确：使用配置文件
from config.config import config
epochs = config.training.num_epochs
batch_size = config.training.batch_size

# ❌ 错误：硬编码超参数
epochs = 20
batch_size = 32
```

#### 5.3 模型保存规范
- 模型权重: `data/models/best_{model_name}_model.pth`
- 实验结果: `data/results/{model_name}_results.pkl`
- 自动保存最佳验证性能的模型

### 6. 实验管理

#### 6.1 结果记录
每次实验必须记录：
```python
results = {
    'model_name': model_name,
    'model_info': model.get_model_info(),
    'training_results': training_results,
    'test_results': test_results,
    'hyperparameters': {
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs': epochs
    }
}
```

#### 6.2 性能对比
- 使用`analyze_models.py`进行模型对比
- 生成统一格式的性能报告
- 包含准确率、F1分数、推理时间、参数量

---

## 🐛 调试与测试规则

### 7. 调试流程

#### 7.1 问题诊断顺序
1. **环境检查**: `python quick_test.py`
2. **数据检查**: 确认数据集文件存在
3. **模型检查**: 验证模型创建和前向传播
4. **训练检查**: 使用`--quick`模式快速验证

#### 7.2 常见问题解决

**数据加载失败**:
```python
# 检查数据文件
import os
print("数据文件存在:", os.path.exists('./data/processed/small_dga_dataset.pkl'))
print("备用文件存在:", os.path.exists('./data/small_dga_dataset.pkl'))
```

**CUDA内存不足**:
```python
# 减小batch_size
config.training.batch_size = 16  # 从32减少到16

# 或使用CPU
device = torch.device('cpu')
```

**模型不收敛**:
```python
# 检查学习率
config.training.learning_rate = 0.0001  # 降低学习率

# 检查梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 8. 测试规范

#### 8.1 单元测试
```python
# ✅ 模型创建测试
def test_model_creation():
    model = CNNModel(vocab_size=40, d_model=64)
    assert model is not None
    assert hasattr(model, 'forward')

# ✅ 前向传播测试
def test_forward_pass():
    model = CNNModel(vocab_size=40)
    x = torch.randint(0, 40, (4, 20))
    output = model(x)
    assert output.shape == (4, 2)
```

#### 8.2 集成测试
- 使用`quick_test.py`进行完整性测试
- 使用`--quick`模式进行快速训练测试
- 验证所有模型的训练和推理流程

---

## 📝 代码质量规则

### 9. 编码规范

#### 9.1 命名规范
```python
# ✅ 类名：大驼峰
class MambaModel(BaseModel):
    pass

# ✅ 函数名：小写+下划线
def create_data_loaders():
    pass

# ✅ 变量名：小写+下划线
model_name = 'cnn'
batch_size = 32

# ✅ 常量：大写+下划线
DEFAULT_VOCAB_SIZE = 40
```

#### 9.2 文档规范
```python
# ✅ 类文档
class MambaModel(BaseModel):
    """Mamba模型用于DGA检测
    
    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_layers: Mamba层数
    """

# ✅ 函数文档
def train_model(model_name: str, quick_test: bool = False):
    """训练指定模型
    
    Args:
        model_name: 模型名称
        quick_test: 是否快速测试模式
        
    Returns:
        训练结果字典
    """
```

#### 9.3 错误处理
```python
# ✅ 正确的错误处理
try:
    dataset = load_dataset(dataset_path)
except FileNotFoundError as e:
    print(f"❌ 数据集文件不存在: {e}")
    print("请确保数据集文件存在，或先运行数据预处理脚本")
    return None
except Exception as e:
    print(f"❌ 加载数据集时发生错误: {e}")
    return None
```

### 10. 性能优化

#### 10.1 内存优化
```python
# ✅ 使用生成器减少内存占用
def data_generator():
    for batch in data_loader:
        yield batch

# ✅ 及时删除不需要的变量
del large_tensor
torch.cuda.empty_cache()  # 清理GPU缓存
```

#### 10.2 计算优化
```python
# ✅ 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 🔄 版本控制规则

### 11. Git工作流

#### 11.1 分支管理
- `main`: 稳定版本
- `develop`: 开发版本
- `feature/model-name`: 新模型开发
- `fix/issue-description`: 问题修复

#### 11.2 提交规范
```bash
# ✅ 提交信息格式
git commit -m "feat: 添加Mamba模型实现"
git commit -m "fix: 修复数据加载路径问题"
git commit -m "docs: 更新README文档"
git commit -m "refactor: 重构训练流程"
```

#### 11.3 忽略文件
```gitignore
# 模型文件
*.pth
*.pkl

# 数据文件
data/raw/
data/processed/

# 日志文件
logs/
*.log

# Python缓存
__pycache__/
*.pyc

# 实验结果
results/
figures/
```

---

## 📈 性能监控规则

### 12. 监控指标

#### 12.1 必需指标
- **准确率 (Accuracy)**: 分类正确的样本比例
- **F1分数**: 精确率和召回率的调和平均
- **推理时间**: 单样本平均推理时间
- **参数量**: 模型总参数数量
- **内存使用**: 训练和推理时的内存占用

#### 12.2 性能基准
```python
# 当前项目基准（截至目前）
BENCHMARKS = {
    'CNN': {'accuracy': 0.9444, 'f1': 0.9443, 'inference_ms': 2.66, 'params': 195000},
    'LSTM': {'accuracy': 0.9528, 'f1': 0.9528, 'inference_ms': 9.55, 'params': 689000},
    'MoE': {'accuracy': 0.9444, 'f1': 0.9443, 'inference_ms': 9.17, 'params': 888000}
}
```

### 13. 实验追踪

#### 13.1 实验记录
每次实验需记录：
- 模型配置和超参数
- 训练时间和收敛情况
- 最终性能指标
- 硬件环境信息

#### 13.2 结果对比
```python
# ✅ 使用统一的结果对比脚本
python analyze_models.py --chart
python main.py analyze --chart
```

---

## 🚨 安全与稳定性规则

### 14. 数据安全

#### 14.1 敏感信息保护
- 不在代码中硬编码路径和配置
- 使用配置文件管理敏感参数
- 避免在日志中输出敏感信息

#### 14.2 模型安全
```python
# ✅ 安全的模型加载
def load_model_safely(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        return state_dict
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")
```

### 15. 稳定性保证

#### 15.1 异常处理
- 所有外部依赖都要有异常处理
- 提供友好的错误信息
- 实现优雅的降级机制

#### 15.2 资源管理
```python
# ✅ 正确的资源管理
with torch.cuda.device(device_id):
    # GPU操作
    pass

# 训练结束后清理
torch.cuda.empty_cache()
```

---

## 📚 文档维护规则

### 16. 文档更新

#### 16.1 必需文档
- `README.md`: 项目概述和快速开始
- `DEVELOPMENT_RULES.md`: 本开发规则文档
- 模型文档: 每个新模型的技术文档
- API文档: 核心接口的使用说明

#### 16.2 文档同步
- 代码变更时同步更新文档
- 新增模型时更新性能对比表
- 定期检查文档的准确性

---

## 🎯 总结

### 核心原则
1. **模块化**: 保持清晰的模块边界
2. **一致性**: 统一的接口和规范
3. **可测试**: 完善的测试覆盖
4. **可维护**: 清晰的代码和文档
5. **高性能**: 持续的性能优化

### 开发流程
1. **需求分析** → 确定模型需求和性能目标
2. **设计实现** → 遵循架构规范实现模型
3. **测试验证** → 完整的功能和性能测试
4. **性能调优** → 达到或超越基准性能
5. **文档更新** → 同步更新相关文档
6. **代码审查** → 确保代码质量
7. **集成部署** → 集成到主分支

### 质量保证
- 所有新模型必须通过`quick_test.py`测试
- 性能必须达到基准要求
- 代码必须符合规范
- 文档必须完整准确

---

**遵循这些规则，确保DGA检测项目的高质量、高性能和可持续发展！** 🚀