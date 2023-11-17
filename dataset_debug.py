"""
MT3 baseline training. 
To use random order, use `dataset.dataset_2_random`. Or else, use `dataset.dataset_2`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
import os
import math

import hydra
from tasks.mt3_net import MT3Net


@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    val_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.val),
        **cfg.dataloader.val,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn)
    )    

    batch = next(iter(val_loader))      

if __name__ == "__main__":   
    main()
