from typing import Union, Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np

import abc
from tqdm import tqdm
from pathlib import Path
import wandb

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torch.utils.data import Dataset, DataLoader

@dataclass_json
@dataclass
class TrainerConfig:
    # optimization parameters
    max_epochs: int = 5000
    batch_size: int = 64

    # optimizer
    learning_rate: float = 1e-4
    # (epoch number, multiplier: after this epoch multiplier will be changes to the given number)
    lr_schedule: List[Tuple[int, float]] = None
    betas: Tuple[float, float] = (0.9, 0.999)

    def __post_init__(self):
        # None is not consistent in hashing
        if self.lr_schedule is None:
            object.__setattr__(self, 'lr_schedule', [])


@dataclass_json
@dataclass
class LogConfig:
    num_model_watches: int = 10
    log_every_n_epoch: int = 1

@dataclass_json
@dataclass
class ModelCfBase:
    pass

class TrainerModel(nn.Module, abc.ABC):

    def __init__(self, model_config: ModelCfBase = None):
        super(TrainerModel, self).__init__()
        if model_config is None:
            model_config = ModelCfBase()
        self._conf = model_config


    @abc.abstractmethod
    def configure_optimizer(self, config: TrainerConfig, **kwargs) -> Optimizer:
        pass

    @abc.abstractmethod
    def get_loss(self, *tensor, **kwargs) -> torch.Tensor:
        pass

    @property
    def config(self) -> ModelCfBase:
        return self._conf

class Trainer:
    """Simple Trainer for normal architectures"""

    def __init__(self,
                 model: TrainerModel,
                 train_dataset: Dataset,
                 valid_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 config: TrainerConfig = None,
                 log_config: LogConfig = None):


        self.model = model

        if config is None:
            self.config = TrainerConfig()
        else:
            self.config = config

        if log_config is None:
            self.log_config = LogConfig()
        else:
            self.log_config = log_config

        self.ckpt_path = Path(wandb.run.dir) / 'model.ckpt'


        # take over whatever gpus are on the system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.data_loaders = dict(
            train=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        )
        if valid_dataset:
            self.data_loaders['valid'] = DataLoader(valid_dataset, batch_size=len(valid_dataset))
        if test_dataset:
            self.data_loaders['test'] = DataLoader(test_dataset, batch_size=len(test_dataset))


    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        print(f'saving ckpt in {str(self.ckpt_path)}')
        torch.save(self.model.state_dict(), self.ckpt_path)
        wandb.save(str(self.ckpt_path))


    def train(self, optim_kw: Optional[Dict[str, Any]] = None,
              loss_kw: Optional[Dict[str, Any]] = None):
        optim_kw = optim_kw or {}

        config = self.config
        optim = self.model.configure_optimizer(config, **optim_kw)

        best_loss = float('inf')
        ret = {}

        has_test = 'test' in self.data_loaders
        has_valid = 'valid' in self.data_loaders

        # lr decay
        if config.lr_schedule:
            schedule_gen = iter(config.lr_schedule)
            next_epoch, next_mult = next(schedule_gen)
        else:
            schedule_gen = None
            next_epoch, next_mult = float('inf'), 1

        pbar = tqdm(range(1, config.max_epochs), total=config.max_epochs)
        wandb.watch(self.model, log='all', log_freq=config.max_epochs // 10)
        for epoch in pbar:

            # lr decay
            if epoch > next_epoch:
                lr = config.learning_rate * next_mult
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
                try:
                    next_epoch, next_mult = next(schedule_gen)
                except StopIteration:
                    next_epoch = float('inf')

            test_loss = self._run_epoch('test', loss_kw=loss_kw) if has_test else None
            valid_loss = self._run_epoch('valid', loss_kw=loss_kw) if has_valid else None
            train_loss = self._run_epoch('train', optim=optim, loss_kw=loss_kw)

            ret.update(train_loss=train_loss, test_loss=test_loss, valid_loss=valid_loss)

            if has_valid and valid_loss < best_loss:
                best_loss = valid_loss
                ret.update(test_loss_at_best_epoch=test_loss, best_valid_loss=best_loss,
                           best_valid_epoch=epoch)
                self.save_checkpoint()

            if epoch % self.log_config.log_every_n_epoch == 0:
                wandb.log(ret)

            if has_valid:
                pbar.set_description(f"[epoch {epoch}] train_loss = {train_loss:.6f}, "
                                     f"valid_loss = {valid_loss:.6f}")
            else:
                pbar.set_description(f"[epoch {epoch}] train_loss = {train_loss:.6f}")

        ckpt_file = wandb.restore(str(self.ckpt_path))
        self.model.load_state_dict(torch.load(ckpt_file.name))


    def _run_epoch(self, mode: str, optim: Optimizer = None,
                   loss_kw: Optional[Dict[str, Any]] = None):
        loss_kw = loss_kw or {}

        model, config = self.model, self.config
        loader = self.data_loaders[mode]
        is_train = mode == 'train'
        model.train(is_train)

        losses = []
        for it, data_batch in enumerate(loader):

            # place data on the correct device
            x = data_batch[0].to(self.device)
            y = data_batch[1].to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                loss = model.get_loss(x, y, **loss_kw)

            if is_train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses.append(loss.item())

        loss_avg = float(np.mean(losses))

        return loss_avg