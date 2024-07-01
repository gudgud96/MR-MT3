from torch.optim import AdamW
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import T5Config
from tasks.mt3_net_segmem_v2_with_prev import MT3NetSegMemV2WithPrev
from utils import get_cosine_schedule_with_warmup


class MT3NetSegMemV2WithPrevFineTune(MT3NetSegMemV2WithPrev):
    def __init__(self, config, optim_cfg, eval_cfg=None):
        super().__init__(
            config=config, 
            optim_cfg=optim_cfg,
            eval_cfg=eval_cfg
        )

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), self.optim_cfg.lr)