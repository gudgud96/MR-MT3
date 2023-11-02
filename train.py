"""
MT3 baseline training. 
To use random order, use `dataset.dataset_2_random`. Or else, use `dataset.dataset_2`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import shutil
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.dataset_2_random import SlakhDataset, collate_fn
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
import os
pl.seed_everything(365)
import hydra


@hydra.main(config_path="config", config_name="config")
# def main(config, model_config, result_dir, mode, path):
def main(cfg):
    model = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    logger = TensorBoardLogger(save_dir='.',
                               name=cfg.model_type)
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

        train_loader = DataLoader(
            SlakhDataset(**cfg.data.train), 
            **cfg.dataloader.train,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            SlakhDataset(**cfg.data.val), 
            **cfg.dataloader.val,
            collate_fn=collate_fn
        )

        trainer.fit(model, train_loader, val_loader)

    else:
        # TODO: still haven't apply hydra to this part yet
        model = MT3Net.load_from_checkpoint(
            cfg.path,
            config=cfg
        )
        model.eval()
        dic = {}
        for key in model.state_dict():
            if "model." in key:
                dic[key.replace("model.", "")] = model.state_dict()[key]
            else:
                dic[key] = model.state_dict()[key]
        torch.save(dic, cfg.path.replace(".ckpt", ".pth"))   # TODO: need to specify save .pth


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
