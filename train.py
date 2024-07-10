"""
MT3 baseline training. 
To use random order, use `dataset.dataset_2_random`. Or else, use `dataset.dataset_2`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
import os

import hydra
from tasks.mt3_net import MT3Net


@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    # set seed to ensure reproducibility
    pl.seed_everything(cfg.seed)

    model = hydra.utils.instantiate(
        cfg.model, 
        optim_cfg=cfg.optim,
        eval_cfg=cfg.eval    
    )
    logger = TensorBoardLogger(save_dir='.',
                               name=f"{cfg.model_type}_{cfg.dataset_type}")
    
    # sanity check to make sure the correct model is used
    assert cfg.model_type == cfg.model._target_.split('.')[-1], f"Model type mismatch: {cfg.model_type} {cfg.model._target_.split('.')[-1]}"

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)
    tqdm_callback = TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, tqdm_callback],
        **cfg.trainer
    )

    train_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.train),
        # SlakhDataset(**cfg.data.train), 
        **cfg.dataloader.train,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn)
    )

    val_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.val),
        **cfg.dataloader.val,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn)
    )

    if cfg.path is not None and cfg.path != "":
        if cfg.path.endswith(".ckpt"):
            print(f"Validating on {cfg.path}...")
            trainer.validate(
                model, 
                val_loader,
                ckpt_path=cfg.path
            )
            print("Training start...")
            trainer.fit(
                model, 
                train_loader, 
                val_loader,
                ckpt_path=cfg.path
            )

        elif cfg.path.endswith(".pth"):
            print(f"Loading weights from {cfg.path}...")
            model.model.load_state_dict(
                torch.load(cfg.path),
                strict=False
            )
            trainer.validate(
                model, 
                val_loader,
            )
            print("Training start...")
            trainer.fit(
                model, 
                train_loader, 
                val_loader,
            )
        
        else:
            raise ValueError(f"Invalid extension for path: {cfg.path}")
    
    else:
        trainer.fit(
                model, 
                train_loader, 
                val_loader,
            )    

    # save the model in .pt format
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(current_dir, f"{cfg.model_type}_{cfg.dataset_type}", "version_0/checkpoints/last.ckpt")
    model.eval()
    dic = {}
    for key in model.state_dict():
        if "model." in key:
            dic[key.replace("model.", "")] = model.state_dict()[key]
        else:
            dic[key] = model.state_dict()[key]
    torch.save(dic, ckpt_path.replace(".ckpt", ".pt"))
    print(f"Saved model in {ckpt_path.replace('.ckpt', '.pt')}.")
        

if __name__ == "__main__":   
    main()
