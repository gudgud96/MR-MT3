from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5_segmem_v2 import T5SegMemV2
from utils import get_cosine_schedule_with_warmup
from tasks.mt3_base import MT3Base


class MT3NetSegMemV2(MT3Base):
    def __init__(self, config, optim_cfg, eval_cfg=None):
        super().__init__(config, optim_cfg, eval_cfg=eval_cfg)
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5SegMemV2(
            config=T5config,
            segmem_num_layers=self.config.segmem_num_layers,
            segmem_length=self.config.segmem_length,
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        lm_logits = self.forward(inputs=inputs, labels=targets)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), targets.view(-1)
            )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        lm_logits = self.forward(inputs=inputs, labels=targets)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), targets.view(-1)
            )
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.optim_cfg.lr)
        warmup_step = int(self.optim_cfg.warmup_steps)
        print('warmup step: ', warmup_step)
        schedule = {
            'scheduler': get_cosine_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=warmup_step, 
                num_training_steps=self.optim_cfg.num_steps_per_epoch * self.optim_cfg.num_epochs,
                min_lr=self.optim_cfg.min_lr
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [schedule]

        # we follow MT3 to use fixed learning rate
        # NOTE: we find this to not work :(
        # return AdamW(self.model.parameters(), self.config.lr)