import os
import torch.distributed as dist
from config.config import DISTRIBUTED_CONFIG

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = DISTRIBUTED_CONFIG['master_addr']
    os.environ['MASTER_PORT'] = DISTRIBUTED_CONFIG['master_port']
    dist.init_process_group(DISTRIBUTED_CONFIG['backend'], rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def dpo_loss(preferred_outputs, non_preferred_outputs):
    return -torch.log(torch.sigmoid(preferred_outputs - non_preferred_outputs)).mean()