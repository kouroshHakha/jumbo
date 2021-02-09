from typing import List, Tuple, Union


from dataclasses import dataclass
from dataclasses_json import dataclass_json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from gpytorch.mlls import ExactMarginalLogLikelihood


@dataclass_json
@dataclass
class GPTrainConfig:
    max_epochs: int = 100
    learning_rate: float = 1e-2
    aux_lr: float =  1e-3
    lr_schedule: List[Tuple[int, float]] = None
    finetune_all: bool = False
    do_optimize: bool = True

    def __post_init__(self):
        # None is not consistent in hashing
        if self.lr_schedule is None:
            object.__setattr__(self, 'lr_schedule', [])


def train(train_x, train_y, model, train_cf: GPTrainConfig, valid_x=None, valid_y=None,
          test_x=None, test_y=None, log_dir: Union[str, Path]=None, std_y=None,
          verbose: bool = False):

    if log_dir:
        log_dir = Path(log_dir)
        writer = SummaryWriter(log_dir)
        ckpt_path = log_dir / 'ckpt'
    else:
        writer = None
        ckpt_path = ''

    use_valid = valid_x is not None and valid_y is not None
    use_test = test_x is not None and test_y is not None

    if use_valid and log_dir is None:
        raise ValueError('validation set is given but log_dir is None, please provide log_dir')
    if use_test and log_dir is None:
        raise ValueError('test set is given but log_dir is None, please provide log_dir')

    mll: torch.nn.Module = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)

    optim = model.configure_optimizer(train_cf)
    # model_params = model.get_trainable_params()
    # optim = Adam(model_params, lr=train_cf.learning_rate)

    if verbose:
        pbar = tqdm(range(1, train_cf.max_epochs + 1), total=train_cf.max_epochs)
    else:
        pbar = range(1, train_cf.max_epochs + 1)

    best_valid_loss, best_valid_epoch = float('inf'), train_cf.max_epochs

    # lr decay
    if train_cf.lr_schedule:
        schedule_gen = iter(train_cf.lr_schedule)
        next_epoch, next_mult = next(schedule_gen)
    else:
        schedule_gen = None
        next_epoch, next_mult = float('inf'), 1

    output_ll = model(train_x)
    train_nll, _, _ = compute_losses(mll, output_ll, train_y, std_y=std_y)
    for epoch in pbar:
        # lr decay
        if epoch > next_epoch:
            lr = train_cf.learning_rate * next_mult
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            try:
                next_epoch, next_mult = next(schedule_gen)
            except StopIteration:
                next_epoch = float('inf')

        model.train()
        optim.zero_grad()
        output_ll = model(train_x)
        train_nll, _, _ = compute_losses(mll, output_ll, train_y, std_y=std_y)
        train_nll.backward()
        optim.step()


        if writer:
            model.eval()
            output_ll = model(train_x)
            _, train_mse, train_el = compute_losses(mll, output_ll, train_y)
            writer.add_scalar('train/nll', train_nll.item(), global_step=epoch)
            writer.add_scalar('train/mse', train_mse.item(), global_step=epoch)
            writer.add_scalar('train/expected_loss', train_el.item(), global_step=epoch)
            writer.add_scalar('lr', optim.param_groups[0]['lr'], global_step=epoch)

        if use_test and writer is not None:
            model.eval()
            output_ll = model(test_x)
            test_nll, test_mse, test_el = compute_losses(mll, output_ll, test_y, std_y=std_y)
            writer.add_scalar('test/nll', test_nll.item(), global_step=epoch)
            writer.add_scalar('test/mse', test_mse.item(), global_step=epoch)
            writer.add_scalar('test/expected_loss', test_el.item(), global_step=epoch)

        if use_valid:
            model.eval()
            output_ll = model(valid_x)

            valid_nll, valid_mse, valid_el = compute_losses(mll, output_ll, valid_y, std_y=std_y)

            valid_frac = len(valid_x) / (len(valid_x) + len(train_x))
            criteria_loss = valid_frac * valid_nll.item() + (1 - valid_frac) * train_nll.item()
            if writer:
                writer.add_scalar('valid/nll', valid_nll.item(), global_step=epoch)
                writer.add_scalar('valid/mse', valid_mse.item(), global_step=epoch)
                writer.add_scalar('valid/expected_loss', valid_el.item(), global_step=epoch)
                writer.add_scalar('valid/criteria_loss', criteria_loss, global_step=epoch)

            if criteria_loss < best_valid_loss:
                best_valid_loss = criteria_loss
                best_valid_epoch = epoch
                if verbose:
                    print(f'saving ckpt in {str(ckpt_path)}')
                torch.save(model.state_dict(), ckpt_path)

            if verbose:
                pbar.set_description(f"epoch {epoch+1}: train loss {train_nll.item():.5f},"
                                     f"valid loss {valid_nll.item():.5f}")
        else:
            if verbose:
                pbar.set_description(f"epoch {epoch+1}: train loss {train_nll.item():.5f}")
    return best_valid_loss, best_valid_epoch

def compute_losses(mll, output_ll, targets, std_y=None):
    nll = -mll(output_ll, targets)
    mean, var = output_ll.mean, output_ll.variance

    mse = ((mean - targets)**2).mean(0)
    expected_loss = ((mean - targets)**2 + var).mean(0)

    if std_y:
        mse *= std_y ** 2
        expected_loss *= std_y ** 2

    return nll, mse.log10(), expected_loss.log10()