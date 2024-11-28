import torch

# 模型配置
MODEL_CONFIG = {
    'hidden_dim': 256,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1
}

# 训练配置
TRAIN_CONFIG = {
    'short_window': 48,
    'long_window': 256,
    'batch_size': 32,
    'accumulation_steps': 4,
    'lr_inner': 0.01,
    'lr_outer': 1e-5,
    'n_inner_steps': 3,
    'n_outer_steps': 6
}

# 适应配置
ADAPT_CONFIG = {
    'n_adapt_steps': 5,
    'lr_adapt': 0.01
}

# 分布式配置
DISTRIBUTED_CONFIG = {
    'master_addr': 'localhost',
    'master_port': '12355',
    'backend': 'nccl'
}

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')