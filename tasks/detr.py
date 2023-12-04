from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from models.t5_detr import T5DETR
from utils import get_cosine_schedule_with_warmup
import pandas as pd

class DETR(pl.LightningModule):

    def __init__(self, config, optim_cfg):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        T5config = T5Config.from_dict(OmegaConf.to_container(self.config))
        self.model: nn.Module = T5DETR(T5config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # input.shape = (B, framesize, n_mels) = (B, 256, 512)
        # targets.shape = (B, n_tokens, 4)
        lm_logits = self.forward(inputs=inputs, labels=targets)

        # creating a mask for the "not found" class
        negative_mask = (targets<0)
        outofbounds_mask = (targets>self.model.config.vocab_size_onset)

        # breaking the mask into 4 parts
        pitch_mask = negative_mask[:,:,0]
        program_mask = negative_mask[:,:,1]
        onset_mask = torch.logical_or(negative_mask[:,:,2], outofbounds_mask[:,:,2])
        offset_mask = torch.logical_or(negative_mask[:,:,3], outofbounds_mask[:,:,3])

        # converting negative classes to its "not found" class number
        targets[:,:,0][pitch_mask] = self.model.config.vocab_size_pitch
        targets[:,:,1][program_mask] = self.model.config.vocab_size_program
        targets[:,:,2][onset_mask] = self.model.config.vocab_size_onset
        targets[:,:,3][offset_mask] = self.model.config.vocab_size_offset

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            pitch_loss = loss_fct(
                lm_logits[0].flatten(0,1), targets[:,:,0].flatten(0,1)
            )
            program_loss = loss_fct(
                lm_logits[1].flatten(0,1), targets[:,:,1].flatten(0,1)
            )
            start_loss = loss_fct(
                lm_logits[2].flatten(0,1), targets[:,:,2].flatten(0,1)
            )
            end_loss = loss_fct(
                lm_logits[3].flatten(0,1), targets[:,:,3].flatten(0,1)
            )
        total_loss = pitch_loss + program_loss + start_loss #+ end_loss               
        self.log('train/pitch_loss', pitch_loss, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/program_loss', program_loss, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/onset_loss', start_loss, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/offset_loss', end_loss, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/loss', total_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        if batch_idx == 0:
            if self.current_epoch == 0:
                self._log_text(targets, "train/targets", max_sentences=4, logger=self)
            # get top1 predictions out of the lm_logits tuples
            pitch_preds = lm_logits[0].argmax(-1)
            program_preds = lm_logits[1].argmax(-1)
            onset_preds = lm_logits[2].argmax(-1)
            offset_preds = lm_logits[3].argmax(-1)
            pred = torch.stack((pitch_preds, program_preds, onset_preds, offset_preds), dim=2)
            self._log_text(pred, "train/preds", max_sentences=4, logger=self)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch 
        # input.shape = (B, framesize, n_mels) = (B, 256, 512)
        # targets.shape = (B, n_tokens, 4)
        lm_logits = self.forward(inputs=inputs, labels=targets)

        # creating a mask for the "not found" class
        negative_mask = (targets<0)
        outofbounds_mask = (targets>self.model.config.vocab_size_onset)

        # breaking the mask into 4 parts
        pitch_mask = negative_mask[:,:,0]
        program_mask = negative_mask[:,:,1]
        onset_mask = torch.logical_or(negative_mask[:,:,2], outofbounds_mask[:,:,2])
        offset_mask = torch.logical_or(negative_mask[:,:,3], outofbounds_mask[:,:,3])

        # converting negative classes to its "not found" class number
        targets[:,:,0][pitch_mask] = self.model.config.vocab_size_pitch
        targets[:,:,1][program_mask] = self.model.config.vocab_size_program
        targets[:,:,2][onset_mask] = self.model.config.vocab_size_onset
        targets[:,:,3][offset_mask] = self.model.config.vocab_size_offset

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            pitch_loss = loss_fct(
                lm_logits[0].flatten(0,1), targets[:,:,0].flatten(0,1)
            )
            program_loss = loss_fct(
                lm_logits[1].flatten(0,1), targets[:,:,1].flatten(0,1)
            )
            start_loss = loss_fct(
                lm_logits[2].flatten(0,1), targets[:,:,2].flatten(0,1)
            )
            end_loss = loss_fct(
                lm_logits[3].flatten(0,1), targets[:,:,3].flatten(0,1)
            )


        total_loss = pitch_loss + program_loss + start_loss #+ end_loss               
        self.log('val/pitch_loss', pitch_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/program_loss', program_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/onset_loss', start_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/offset_loss', end_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            if self.current_epoch == 0:
                self._log_text(targets, "val/targets", max_sentences=4, logger=self)
            pitch_preds = lm_logits[0].argmax(-1)
            program_preds = lm_logits[1].argmax(-1)
            onset_preds = lm_logits[2].argmax(-1)
            offset_preds = lm_logits[3].argmax(-1)
            pred = torch.stack((pitch_preds, program_preds, onset_preds, offset_preds), dim=2)
            self._log_text(pred, "val/preds", max_sentences=4, logger=self)
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
                    pitch = token[0].item()
                    program = token[1].item()
                    onset = token[2].item()
                    offset = token[3].item()
                    token_str = token_str + \
                    f"<{pitch}, {program}, {onset}, {offset}>" + ', '
                plugin_list.append(token_str)        
        s = pd.Series(plugin_list, name="token sequence")
        logger.logger.experiment.add_text(tag, s.to_markdown(), global_step=self.global_step)
