"""
MT3 baseline training. 
To use random order, use `dataset.dataset_2_random`. Or else, use `dataset.dataset_2`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from torch.optim import AdamW
from omegaconf import OmegaConf
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Config
from dataset.dataset_2_random import SlakhDataset, collate_fn
from torch.utils.data import DataLoader
from models.t5 import T5ForConditionalGeneration
import torch.nn as nn
from utils import get_cosine_schedule_with_warmup, get_result_dir
import torch
import json
import pytorch_lightning as pl
import os
pl.seed_everything(365)
import copy
import argparse
import mlflow
import datetime
import hydra


class MT3Net(pl.LightningModule):

    def __init__(self, config, result_dir='./results'):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config.model))
        self.model: nn.Module = T5ForConditionalGeneration(T5config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs=inputs, labels=targets)
        self.log('train_loss', outputs.loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        mlflow.log_metric("train_loss", outputs.loss)
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs=inputs, labels=targets)
        self.log('val_loss', outputs.loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        mlflow.log_metric("val_loss", outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.lr)
        warmup_step = int(self.config.warmup_steps)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=1289*self.config.num_epochs,
                min_lr=5e-5
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]

        # we follow MT3 to use fixed learning rate
        # NOTE: we find this to not work :(
        # return AdamW(self.model.parameters(), self.config.lr)

    def train_dataloader(self):
        dataset = SlakhDataset(**self.config.data.train)
        train_loader = DataLoader(
            dataset, 
            **self.config.dataloader.train,
            collate_fn=collate_fn
        )
        return train_loader

    def val_dataloader(self):
        dataset = SlakhDataset(**self.config.data.val)
        val_loader = DataLoader(
            dataset, 
            **self.config.dataloader.val,
            collate_fn=collate_fn
        )
        return val_loader

@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    model = MT3Net(cfg)
    logger = TensorBoardLogger(save_dir='.',
                               name='dummy_run')
    # TODO: use config file name 
    # https://stackoverflow.com/a/70070004/9793316

    num_epochs = int(cfg.num_epochs)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)
    tqdm_callback = TQDMProgressBar(refresh_rate=1)

    if cfg.mode == "train":
        trainer = pl.Trainer(
            logger=logger,
            # default_root_dir=os.path.join(os.getcwd(), 'logs'),
            default_root_dir='.',
            callbacks=[lr_monitor, checkpoint_callback, tqdm_callback],
            **cfg.trainer
        )
        trainer.fit(model)

    else:
        # TODO: still haven't apply hydra to this part yet
        model = MT3Net.load_from_checkpoint(
            path,
            config=config,
            model_config=model_config,
            result_dir=result_dir
        )
        model.eval()
        dic = {}
        for key in model.state_dict():
            if "model." in key:
                dic[key.replace("model.", "")] = model.state_dict()[key]
            else:
                dic[key] = model.state_dict()[key]
        torch.save(dic, path.replace(".ckpt", ".pth"))   # TODO: need to specify save .pth


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', default="train")      # option that takes a value
    # parser.add_argument('--path')      # option that takes a value
    # args = parser.parse_args()

    # conf_file = 'config/config.yaml'
    # model_config = 'config/mt3_config.json'
    # print(f'Config {conf_file}')
    # conf = OmegaConf.load(conf_file)
    
    # result_dir = ""
    # if args.mode == "train":
    #     datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     result_dir = f"logs/results_norm_randomorder_{datetime_str}"
    #     print('Creating: ', result_dir)
    #     os.makedirs(result_dir, exist_ok=False)
    #     shutil.copy(conf_file, f'{result_dir}/config.yaml')
    
    main()
