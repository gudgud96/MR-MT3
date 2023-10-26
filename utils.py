from collections import OrderedDict
import math
from torch.optim.lr_scheduler import LambdaLR
import torch


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_steps, model_dim, last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        super(NoamScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self, epoch=None):
        cur_step = self.last_epoch + 2
        cur_lr = 0.002 * self.model_dim ** (0.5) * min(cur_step ** (-0.5), cur_step * self.warmup_steps ** (-1.5))
        return [cur_lr for group in self.optimizer.param_groups]


def get_noam_scheduler(optimizer, warmup_steps, model_dim):
    return NoamScheduler(optimizer, warmup_steps, model_dim)


def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float = 0.5, 
    last_epoch: int = -1,
    min_lr: float = 2e-5
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            lr = float(current_step) / float(max(1, num_warmup_steps))
            return lr
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        lr = max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return lr
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_mt3_optimizer(
    optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        return min(1, current_step / num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_result_dir(lightning_logs_dir='results'):
    import glob
    log_dir = glob.glob(f'{lightning_logs_dir}/**/config.yaml')
    log_dir_version = list(
        map(lambda x: int(x.split('/')[-2].replace('version_', '')), log_dir))
    if len(log_dir_version) != 0:
        exp_num = '{:0>3d}'.format(max(log_dir_version) + 1)
    else:
        exp_num = '{:0>3d}'.format(1)
    return f'./{lightning_logs_dir}'


def remove_state_dict_prefix(state_dict, prefix='module.'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace(prefix, '')] = v
    return new_state_dict
