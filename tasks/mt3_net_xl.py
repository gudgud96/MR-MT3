from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5_xl import T5WithXLDecoder
from utils import get_cosine_schedule_with_warmup

class MT3NetXL(pl.LightningModule):

    def __init__(self, config, model_config_path='config/mt3_config.json', result_dir='./results'):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config.model))
        self.model: nn.Module = T5WithXLDecoder(T5config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(
            inputs=inputs, 
            labels=targets
        )
        self.log('train_loss', outputs.loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return outputs.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(
            inputs=inputs, 
            labels=targets
        )
        self.log('val_loss', outputs.loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
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