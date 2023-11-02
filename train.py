"""
MT3 baseline training. 
To use random order, use `dataset.dataset_2_random`. Or else, use `dataset.dataset_2`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.dataset_2_random import SlakhDataset, collate_fn
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
import os

import hydra


@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    # set seed to ensure reproducibility
    pl.seed_everything(cfg.seed)

    model = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    logger = TensorBoardLogger(save_dir='.',
                               name=cfg.model_type)
    
    # sanity check to make sure the correct model is used
    assert cfg.model_type == cfg.model._target_.split('.')[-1]

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)
    tqdm_callback = TQDMProgressBar(refresh_rate=1)

    if cfg.mode == "train":
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback, tqdm_callback],
            **cfg.trainer
        )

        train_loader = DataLoader(
            SlakhDataset(**cfg.data.train), 
            **cfg.dataloader.train,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            SlakhDataset(**cfg.data.val), 
            **cfg.dataloader.val,
            collate_fn=collate_fn
        )

        trainer.fit(model, train_loader, val_loader)

    else:
        # To activate this part, run:
        # python train.py mode=test
        model = MT3Net.load_from_checkpoint(
            cfg.path,
            config=cfg
        )
        model.eval()
        dic = {}
        for key in model.state_dict():
            if "model." in key:
                dic[key.replace("model.", "")] = model.state_dict()[key]
            else:
                dic[key] = model.state_dict()[key]
        torch.save(dic, cfg.path.replace(".ckpt", ".pth"))   # TODO: need to specify save .pth


if __name__ == "__main__":   
    main()
