# Models Module

# 只导入基础模型，避免依赖问题
try:
    from .implementations.bilstm_attention_model import LSTMModel
except ImportError:
    LSTMModel = None

try:
    from .implementations.simple_lstm_model import SimpleLSTMModel
except ImportError:
    SimpleLSTMModel = None

try:
    from .implementations.cnn_model import CNNModel
except ImportError:
    CNNModel = None

try:
    from .implementations.tcbam_models import TCBAMModel
except ImportError:
    TCBAMModel = None

# 有依赖问题的模型延迟导入
MambaModel = None
MoEModel = None
SimplifiedMoEModel = None
MambaformerModel = None
HomogeneousMoEModel = None

__all__ = [
    'LSTMModel',
    'SimpleLSTMModel', 
    'CNNModel',
    'TCBAMModel'
]