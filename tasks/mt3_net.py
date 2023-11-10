from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5 import T5ForConditionalGeneration
from utils import get_cosine_schedule_with_warmup
import pandas as pd

class MT3Net(pl.LightningModule):

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

        if batch_idx == 0:
            if self.current_epoch == 0:
                self._log_text(targets, "val/targets", max_sentences=4, logger=self)
            self._log_text(lm_logits.argmax(-1), "val/preds", max_sentences=4, logger=self)
        
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

    def _log_text(self, token_seqs, tag, max_sentences, logger):       
        plugin_list = []
        for idx, token_seq in enumerate(token_seqs):
            if idx < max_sentences: 
                token_str = ''
                for token in token_seq:
                    token_str = token_str + get_token_name(token) + ', '
                plugin_list.append(token_str)        
        s = pd.Series(plugin_list, name="token sequence")
        logger.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.global_step)


class MT3NetCTC(MT3Net):
    def __init__(self, config, optim_cfg):
        super().__init__(config, optim_cfg)
        self.loss_fct = nn.CTCLoss(blank=0, zero_infinity=True) # need to use 2 to avoid Nan
        # https://discuss.pytorch.org/t/ctcloss-predicts-blanks-after-a-few-batches/40190/8

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        lm_logits = self.forward(inputs=inputs, labels=targets)

        if targets is not None:
            # finding the first occurance of eos
            # and use the index to be the target length
            target_lengths = ((targets == 1).cumsum(1).cumsum(1)==1).float().argsort(1)[:,-1]
            input_lengths = ((lm_logits.argmax(-1) == 1).cumsum(1).cumsum(1)==1).float().argsort(1)[:,-1]

            loss = self.loss_fct(
                lm_logits.transpose(0,1).log_softmax(-1),
                targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths
            )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        lm_logits = self.forward(inputs=inputs, labels=targets)

        if targets is not None:
            # finding the first occurance of eos
            # and use the index to be the target length
            target_lengths = ((targets == 1).cumsum(1).cumsum(1)==1).float().argsort(1)[:,-1]
            input_lengths = ((lm_logits.argmax(-1) == 1).cumsum(1).cumsum(1)==1).float().argsort(1)[:,-1]

            loss = self.loss_fct(
                lm_logits.transpose(0,1).log_softmax(-1),
                targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths
            )
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            if self.current_epoch == 0:
                self._log_text(targets, "val/targets", max_sentences=4, logger=self)
            self._log_text(lm_logits.argmax(-1), "val/preds", max_sentences=4, logger=self)
        
        # no need to use it in this stage
        # return loss



class MT3NetWeightedLoss(pl.LightningModule):

    def __init__(self, config, loss_weights, optim_cfg):
        super().__init__()
        self.config = config
        self.loss_weights = loss_weights
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

            pitch_mask = (labels_flatten >= 1001+3) & (labels_flatten <=  1128+3)
            velocity_mask = (labels_flatten >= 1129+3) & (labels_flatten <=  1130+3)
            tie_mask = (labels_flatten == 1131+3)
            instrument_tokens_mask = (labels_flatten >= 1135) & (labels_flatten <= 1262)
            drum_mask = (labels_flatten >= 1260+3) & (labels_flatten <= 1387+3)
            time_mask = (labels_flatten >= 0+3) & (labels_flatten <= 1000+3)
            eos_mask = (labels_flatten == 1)
            

            # pad_mask = labels_flatten != -100
            
            loss_unmasked = loss_fct_raw(
                lm_logits.view(-1, lm_logits.size(-1)), labels_flatten
            )

            loss_pitch = torch.masked_select(loss_unmasked, pitch_mask)
            loss_velocity = torch.masked_select(loss_unmasked, velocity_mask)
            loss_tie = torch.masked_select(loss_unmasked, tie_mask)
            loss_instruments = torch.masked_select(loss_unmasked, instrument_tokens_mask)
            loss_drum = torch.masked_select(loss_unmasked, drum_mask)
            loss_time = torch.masked_select(loss_unmasked, time_mask)
            loss_eos = torch.masked_select(loss_unmasked, eos_mask)

            num_tokens = (loss_pitch.shape[0] + \
                          loss_velocity.shape[0] + \
                          loss_tie.shape[0] + \
                          loss_drum.shape[0] + \
                          loss_instruments.shape[0] + \
                          loss_time.shape[0] + \
                          loss_eos.shape[0])
            
            # combine all losses with pre-defined weights
            loss = (loss_pitch.sum() * self.loss_weights.pitch + \
                    loss_velocity.sum() * self.loss_weights.velocity + \
                    loss_tie.sum() * self.loss_weights.tie + \
                    loss_instruments.sum() * self.loss_weights.instrument +\
                    loss_drum.sum() * self.loss_weights.drum +\
                    loss_time.sum() * self.loss_weights.time +\
                    loss_eos.sum() * self.loss_weights.eos    
                    ) / num_tokens

        self.log('train/loss_pitch', loss_pitch.sum()/loss_pitch.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)
        self.log('train/loss_velocity', loss_velocity.sum()/loss_velocity.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)
        self.log('train/loss_tie', loss_tie.sum()/loss_tie.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)
        self.log('train/loss_inst', loss_instruments.sum()/loss_instruments.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)
        self.log('train/loss_drum', loss_drum.sum()/loss_drum.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)
        self.log('train/loss_time', loss_time.sum()/loss_time.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)
        self.log('train/loss_eos', loss_eos.sum()/loss_eos.shape[0],
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True)

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

            pitch_mask = (labels_flatten >= 1001+3) & (labels_flatten <=  1128+3)
            velocity_mask = (labels_flatten >= 1129+3) & (labels_flatten <=  1130+3)
            tie_mask = (labels_flatten == 1131+3)
            instrument_tokens_mask = (labels_flatten >= 1135) & (labels_flatten <= 1262)
            drum_mask = (labels_flatten >= 1260+3) & (labels_flatten <= 1387+3)
            time_mask = (labels_flatten >= 0+3) & (labels_flatten <= 1000+3)
            eos_mask = (labels_flatten == 1)
            

            # pad_mask = labels_flatten != -100
            
            loss_unmasked = loss_fct_raw(
                lm_logits.view(-1, lm_logits.size(-1)), labels_flatten
            )

            loss_pitch = torch.masked_select(loss_unmasked, pitch_mask)
            loss_velocity = torch.masked_select(loss_unmasked, velocity_mask)
            loss_tie = torch.masked_select(loss_unmasked, tie_mask)
            loss_instruments = torch.masked_select(loss_unmasked, instrument_tokens_mask)
            loss_drum = torch.masked_select(loss_unmasked, drum_mask)
            loss_time = torch.masked_select(loss_unmasked, time_mask)
            loss_eos = torch.masked_select(loss_unmasked, eos_mask)

            num_tokens = (loss_pitch.shape[0] + \
                          loss_velocity.shape[0] + \
                          loss_tie.shape[0] + \
                          loss_drum.shape[0] + \
                          loss_instruments.shape[0] + \
                          loss_time.shape[0] + \
                          loss_eos.shape[0])
            
            # combine all losses with pre-defined weights
            loss = (loss_pitch.sum() * self.loss_weights.pitch + \
                    loss_velocity.sum() * self.loss_weights.velocity + \
                    loss_tie.sum() * self.loss_weights.tie + \
                    loss_instruments.sum() * self.loss_weights.instrument +\
                    loss_drum.sum() * self.loss_weights.drum +\
                    loss_time.sum() * self.loss_weights.time +\
                    loss_eos.sum() * self.loss_weights.eos    
                    ) / num_tokens

        self.log('val/loss_pitch', loss_pitch.sum()/loss_pitch.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_velocity', loss_velocity.sum()/loss_velocity.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_tie', loss_tie.sum()/loss_tie.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_inst', loss_instruments.sum()/loss_instruments.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_drum', loss_drum.sum()/loss_drum.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_time', loss_time.sum()/loss_time.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_eos', loss_eos.sum()/loss_eos.shape[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._log_text(targets, "val/targets", max_sentences=4, logger=self)
            self._log_text(lm_logits.argmax(-1), "val/preds", max_sentences=4, logger=self) 
        
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

    def _log_text(self, token_seqs, tag, max_sentences, logger):       
        plugin_list = []
        for idx, token_seq in enumerate(token_seqs):
            if idx < max_sentences: 
                token_str = ''
                for token in token_seq:
                    token_str = token_str + get_token_name(token) + ', '
                plugin_list.append(token_str)        
        s = pd.Series(plugin_list, name="token sequence")
        logger.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.global_step)   


def get_token_name(token_idx):
    # offset by 3 because of special tokens
    # The special tokens: 0=PAD, 1=EOS, and 2=UNK
    token_idx = int(token_idx-3)
    if token_idx >= 1001 and token_idx <= 1128:
        token = f"pitch_{token_idx - 1001}"
    elif token_idx >= 1129 and token_idx <= 1130:
        token = f"velocity_{token_idx - 1129}"
    elif token_idx >= 1131 and token_idx <= 1131:
        token = "tie"
    elif token_idx >= 1132 and token_idx <= 1259:
        token = f"program_{token_idx - 1132}"
    elif token_idx >= 1260 and token_idx <= 1387:
        token = f"drum_{token_idx - 1260}"
    elif token_idx >= 0 and token_idx < 1000:
        token = f"shift_{token_idx}"
    elif token_idx == 1-3:
        token = f"eos"
    else:
        token = f"invalid_{token_idx}"
    
    return token