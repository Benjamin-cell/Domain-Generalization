import torch
import torch.multiprocessing as mp
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from models.transformer import CustomTransformerModel
from data.dataset import split_few_shot_data
from trainers.meta_trainer import meta_finetune_with_dpo
from trainers.adaptation import evaluate_few_with_adaptation
from utils.distributed import setup, cleanup
from config.config import MODEL_CONFIG, TRAIN_CONFIG
import json


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main_worker(rank, world_size, datasets, data_inference, predict_length):
    try:
        setup(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)

        # Setup logging
        if rank == 0:
            timestamp = get_timestamp()
            log_filename = f'experiment_{timestamp}_{predict_length}_transformer.log'
            logging.basicConfig(
                filename=log_filename,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger()

        # Initialize model
        max_feature_count = max(dataset['features'] for dataset in datasets.values())
        base_model = CustomTransformerModel(
            input_dim=max_feature_count,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            predict_length=predict_length,
            nhead=MODEL_CONFIG['n_heads'],
            num_layers=MODEL_CONFIG['n_layers'],
            dropout=MODEL_CONFIG['dropout'],
            freeze_parameters=True
        ).to(device)

        timestamp = get_timestamp()
        log_filename = f'experiment_{timestamp}_transformer.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger()

        # 加载训练数据集
        datasets = {
            'ETTm1': {'path': r"/home/feiteng/dataset/ETTm1.csv", 'features': 7},
            'Traffic': {'path': r"/home/feiteng/dataset/traffic_2(1).csv", 'features': 862},
            'Electricity': {'path': r"/home/feiteng/dataset/output_b.csv", 'features': 321},
            'Weather': {'path': r"/home/feiteng/dataset/weather.csv", 'features': 21},
            'ETTm2': {'path': r"/home/feiteng/dataset/ETTm2.csv", 'features': 7},
            'Exchange': {'path': r"/home/feiteng/dataset/exchange_rate.csv", 'features': 7},
            'Illness': {'path': r"/home/feiteng/dataset/illness.csv", 'features': 7}
        }

        all_datasets = {}
        for name, info in datasets.items():
            data = pd.read_csv(info['path'], parse_dates=['date'])
            data.set_index('date', inplace=True)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

        # 加载推理数据集
        datainference = {
            'ETTh1': {'path': r"/home/feiteng/inference/ETTh1(1).csv", 'features': 7},
            'ETTh2': {'path': r"/home/feiteng/inference/ETTh2(1).csv", 'features': 7},
        }

        data_inference = {}
        for name, info in datainference.items():
            data = pd.read_csv(info['path'], parse_dates=['date'])
            data.set_index('date', inplace=True)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            data_inference[name] = {
                'test': scaled_data,
                'features': info['features']
            }

    except Exception as e:
        if rank == 0:
            logger.error(f"Error in process {rank}: {str(e)}")
        raise e
    finally:
        cleanup()
        torch.cuda.empty_cache()


def main():
    # ... [Dataset loading logic remains the same as in original]

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs to run, but got", world_size)
        return

    mp.spawn(
        main_worker,
        args=(world_size, all_datasets, data_inference, predict_length),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()