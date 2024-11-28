# Meta-Learning Based Cross-Domain Time Series Forecasting via Preference Optimization

## Overview
This project implements a distributed meta-learning framework for time series forecasting using PyTorch. It features a transformer-based or language model architecture with Direct Preference Optimization (DPO) for meta-learning and fast adaptation to new domains. The implementation supports multi-GPU training through PyTorch's Distributed Data Parallel (DDP).

## Key Features
- Transformer-based time series forecasting
- Distributed training support (Multi-GPU)
- Meta-learning with DPO
- Fast domain adaptation
- Flexible window sizes for short and long-term predictions
- Comprehensive evaluation metrics

## Project Structure
```
project/
├── config/
│   └── config.py           # Configuration settings
├── models/
│   ├── __init__.py
│   ├── positional_encoding.py
│   └── transformer.py      # Model architecture
├── data/
│   ├── __init__.py
│   └── dataset.py         # Dataset implementations
├── utils/
│   ├── __init__.py
│   ├── distributed.py     # Distributed training utilities
│   └── metrics.py         # Evaluation metrics
├── trainers/
│   ├── __init__.py
│   ├── meta_trainer.py    # Meta-learning implementation
│   └── adaptation.py      # Fast adaptation methods
└── main.py               # Main entry point
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- Additional dependencies:
  ```
  torch
  pandas
  numpy
  scikit-learn
  higher
  logging
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [project-directory]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Key configurations can be modified in `config/config.py`:

```python
MODEL_CONFIG = {
    'hidden_dim': 256,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1
}

TRAIN_CONFIG = {
    'short_window': 48,
    'long_window': 256,
    'batch_size': 32,
    'accumulation_steps': 4
}
```

## Usage

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir data
```

**Please put them in the `./data` directory**

### Data Preparation
Prepare your time series data in CSV format with the following structure:
- Required columns: 'date' and feature columns
- Date column should be in datetime format
- All feature columns should be numerical

### Training
To start training:
```bash
python main.py
```

The script will automatically:
1. Initialize distributed training environment
2. Load and preprocess data
3. Set up the model
4. Perform meta-learning
5. Adapt to target domains
6. Save results and model checkpoints

### Distributed Training
The project supports multi-GPU training out of the box:
- Automatically detects available GPUs
- Uses PyTorch's DistributedDataParallel
- Requires at least 2 GPUs for distributed training

## Model Architecture
The model uses a transformer-based architecture with:
- Positional encoding for temporal information
- Multi-head self-attention mechanism
- Configurable number of transformer layers
- Adaptive input and output projections

## Meta-Learning Strategy
The meta-learning process involves:
1. Generating preference pairs across domains
2. DPO-based meta-training
3. Fast adaptation to target domains
4. Performance evaluation on test sets

## Outputs
The training process generates:
- Model checkpoints (saved periodically)
- Training logs
- Evaluation metrics in JSON format
- Performance visualizations

## Customization
### Adding New Datasets
1. Prepare your dataset in CSV format
2. Add dataset path and features count to the configuration
3. The data loader will automatically handle the preprocessing

### Modifying Model Architecture
Modify `models/transformer.py` to customize the model architecture:
- Adjust number of layers
- Change hidden dimensions
- Modify attention mechanisms
- Add new model components

## Performance Monitoring
The system logs:
- Training progress
- Meta-learning loss
- Adaptation metrics
- Final evaluation results

## Acknowledgments
- PyTorch team for the distributed training framework
- [Any other acknowledgments]