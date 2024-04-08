from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5 import T5ForConditionalGeneration
from utils import get_cosine_schedule_with_warmup

class MT3Net(pl.LightningModule):

    def __init__(self, config, optim_cfg):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5ForConditionalGeneration(
            T5config
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    # def on_train_batch_start(self, *args, **kwargs):
    #     self.model.train()
    #     pl.seed_everything(365)

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
        
        # no need to use it in this stage
        # return loss

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


class MT3NetWeightedLoss(pl.LightningModule):

    def __init__(self, config, optim_cfg):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5ForConditionalGeneration(T5config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        lm_logits = self.forward(inputs=inputs, labels=targets)

        if targets is not None:
            loss_fct_raw = nn.CrossEntropyLoss(reduction="none")

            # pitch (1001, 1128)
            # velocity (1129, 1130)
            # tie (1131, 1131)
            # program (1132, 1259)
            # drum (1260, 1387)

            labels_flatten = targets.view(-1)
            instrument_tokens_mask = (labels_flatten >= 1135) & (labels_flatten <= 1262)
            pad_mask = labels_flatten != -100
            
            loss_unmasked = loss_fct_raw(
                lm_logits.view(-1, lm_logits.size(-1)), labels_flatten
            )

            loss_instruments = torch.masked_select(loss_unmasked, instrument_tokens_mask)
            loss_masked = torch.masked_select(loss_unmasked, pad_mask)
            loss = (loss_masked.sum() + 2 * loss_instruments.sum()) / (loss_instruments.shape[0] + loss_masked.shape[0])

        self.log('train_loss_other', loss_masked.sum()/loss_masked.shape[0], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_loss_inst', loss_instruments.sum()/loss_instruments.shape[0], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)            
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        lm_logits = self.forward(inputs=inputs, labels=targets)

        if targets is not None:
            loss_fct_raw = nn.CrossEntropyLoss(reduction="none")

            # pitch (1001, 1128)
            # velocity (1129, 1130)
            # tie (1131, 1131)
            # program (1132, 1259)
            # drum (1260, 1387)

            labels_flatten = targets.view(-1)
            instrument_tokens_mask = (labels_flatten >= 1135) & (labels_flatten <= 1262)
            pad_mask = labels_flatten != -100
            
            loss_unmasked = loss_fct_raw(
                lm_logits.view(-1, lm_logits.size(-1)), labels_flatten
            )

            loss_instruments = torch.masked_select(loss_unmasked, instrument_tokens_mask)
            loss_masked = torch.masked_select(loss_unmasked, pad_mask)
            loss = (loss_masked.sum() + 2 * loss_instruments.sum()) / (loss_instruments.shape[0] + loss_masked.shape[0])

        self.log('val_loss_other', loss_masked.sum()/loss_masked.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss_inst', loss_instruments.sum()/loss_instruments.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # no need to use it in this stage
        # return loss

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

