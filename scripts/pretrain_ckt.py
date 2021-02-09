from typing import Union, Dict, Any

import argparse
import numpy as np
from pathlib import Path
import wandb

import torch
from torch.utils.data import TensorDataset


from jumbo.util import train_test_split, read_pickle, write_yaml
from jumbo.nn.basics import TrainerConfig, Trainer
from jumbo.nn.models import SimpleModel, ModelCf


def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str,
                        help='If provided, collection is skipped and this data is used instead.')
    parser.add_argument('--seed', '-s', default=1, type=int,
                        help='random seed to run in parallel if possible')
    parser.add_argument('--root', '-r', default='pretrain', type=str,
                        help='The root directory for book keeping')
    parser.add_argument('--fract', '-f', default=0.001, type=float,
                        help='fraction of the dataset for low fidel')
    parser.add_argument('--npoints', '-n', default=100, type=int,
                        help='Number of points to collect and train on!')
    parser.add_argument('--max_epochs', default=1000, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning Rate')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='batch size')
    parser.add_argument('--nh_unit', '-nu', default=32, type=int, help='Number of hidden units')
    parser.add_argument('--nh_layer', '-nl', default=3, type=int, help='Number of hidden layers')
    parser.add_argument('--latent_dim', '-ld', default=32, type=int,
                        help='Dimension of latent variable')
    parser.add_argument('--activation', '-act', default='tanh', type=str,
                        help='Ativation for intermediate layers')
    parser.add_argument('--optimize_out', action='store_true',
                        help='If true the output MSE loss will also be minimized')
    parser.add_argument('--skip_conn', action='store_true',
                        help='Use the resnet style (with skip connections) architecture')
    parser.add_argument('--use_bnorm', action='store_true',
                        help='Use BatchNorm for normalizing the output of the previous layer '
                             'before concatenating the input')
    parser.add_argument('--use_resnet', action='store_true',
                        help='Use Residual architecture for better performance?')
    parser.add_argument('--use_huber', action='store_true',
                        help='Use HuberLoss instead of MSE for latent')
    parser.add_argument('--drop_out', type=float, default=0.,
                        help='The probablity of droping activations')
    parser.add_argument('--run_id', type=str, help='If provided wandb will attempt to use this '
                                                   'run_id to resume a previous experiment')


    return parser.parse_args()


def pretrain(rng: Union[np.random.RandomState, int], data: Dict[str, Any],
             pargs: argparse.Namespace):

    x, ynorm, xstat, ystat = data['xarr'], data['ynorm'], data['xstat'], data['ystat']
    input_dim = x.shape[-1]

    xtrain, ytrain, xvalid, yvalid = train_test_split(x, ynorm, rng)
    train_cf = TrainerConfig(
        max_epochs=pargs.max_epochs,
        batch_size=min(pargs.batch_size, len(xtrain)),
        learning_rate=pargs.lr
    )

    train_set = TensorDataset(torch.tensor(xtrain).float(), torch.tensor(ytrain).float())
    valid_set = TensorDataset(torch.tensor(xvalid).float(), torch.tensor(yvalid).float())

    mcf = ModelCf(
        input_dim=input_dim,
        out_dim=ynorm.shape[-1] if ynorm.ndim > 1 else ynorm.shape[0],
        ystat=ystat,
        nhidden=pargs.nh_unit,
        nz_dim=pargs.latent_dim,
        optimize_out=pargs.optimize_out,
        nh_layer=pargs.nh_layer,
        activation=pargs.activation,
        output_act='tanh',
        skip_conn=pargs.skip_conn,
        bnorm=pargs.use_bnorm,
        drop_out=pargs.drop_out,
        use_resnet=pargs.use_resnet,
        use_huber=pargs.use_huber,
    )

    mcf_path = Path(wandb.run.dir) / 'model_config.yaml'
    write_yaml(mcf_path, dict(model_config=mcf.to_dict(), train_config=train_cf.to_dict()))

    model = SimpleModel(mcf)

    trainer = Trainer(model, train_set, valid_set, config=train_cf)
    trainer.train()

def main(pargs: argparse.Namespace):
    rng = np.random.RandomState(pargs.seed)
    torch.manual_seed(pargs.seed)

    data = read_pickle(pargs.data)

    if not _args.no_train:
        pretrain(rng, data, pargs)


if __name__ == '__main__':
    _args = _parse_args()
    global args

    prj_name = 'ckt_test'
    rid = wandb.util.generate_id() if _args.run_id is None else _args.run_id
    wandb.init(project=prj_name, id=rid)
    config = {k: v for k, v in vars(_args).items() if k not in ['run_id']}
    wandb.config.update(config)
    main(_args)