from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import glob
from test import get_scores


class MT3Base(pl.LightningModule):
    """
    Base class for MT3 related experiments
    """
    def __init__(self, config, optim_cfg, eval_cfg=None):
        super().__init__()
        self.config = config
        self.optim_cfg = optim_cfg
        self.eval_cfg = eval_cfg
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    @rank_zero_only
    def on_validation_epoch_end(self):
        if self.current_epoch >= self.eval_cfg.eval_after_num_epoch:
            if self.current_epoch % self.eval_cfg.eval_per_epoch == 0:
                eval_audio_dir = sorted(glob.glob(self.eval_cfg.audio_dir))
                if self.eval_cfg.eval_first_n_examples:
                    eval_audio_dir = eval_audio_dir[:self.eval_cfg.eval_first_n_examples]

                self.model.eval()
                scores = get_scores(
                    model=self.model, 
                    eval_audio_dir=eval_audio_dir,
                    eval_dataset="Slakh",
                    ground_truth_midi_dir=self.eval_cfg.midi_dir,
                    verbose=False
                )

                self.log('val_f1_flat', scores['Onset F1'], on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_f1_midi_class', scores['Onset + program F1 (midi_class)'], on_step=False, on_epoch=True)
                self.log('val_f1_full', scores['Onset + program F1 (full)'], on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        raise NotImplementedError
