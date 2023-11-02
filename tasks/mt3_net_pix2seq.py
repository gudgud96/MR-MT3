"""
MT3 with random order and pix2seq noise training. 
Uses `dataset_2_random_noise`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from torch.optim import AdamW
from omegaconf import OmegaConf
from transformers import T5Config
from dataset.dataset_2_random_noise import SlakhDataset, collate_fn
from torch.utils.data import DataLoader
from models.t5 import T5ForConditionalGeneration
import torch.nn as nn
from utils import get_cosine_schedule_with_warmup, get_result_dir
import torch
import json
import pytorch_lightning as pl
import os
pl.seed_everything(365)


class MT3NetPix2Seq(pl.LightningModule):

    def __init__(self, config, optim_cfg):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5ForConditionalGeneration(T5config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, decoder_inputs, targets = batch
        outputs = self.forward(
            inputs=inputs, 
            decoder_input_ids=decoder_inputs, 
            labels=targets
        )
        self.log('train_loss', outputs.loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, decoder_inputs, targets = batch
        outputs = self.forward(
            inputs=inputs, 
            decoder_input_ids=decoder_inputs, 
            labels=targets
        )
        self.log('val_loss', outputs.loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.optim_cfg.lr)
        warmup_step = int(self.optim_cfg.warmup_steps)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=1289*self.optim_cfg.num_epochs,
                min_lr=5e-5
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]