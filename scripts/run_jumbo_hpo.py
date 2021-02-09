

import argparse

from jumbo.optimize import jumbo_optimize
from hpo.functions import get_eval_function, get_transform

from utils.pdb import register_pdb_hook
register_pdb_hook()


def _arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_path', '-pp', type=str,
                        help='Pretrained path that contains model.pt and config.yaml')
    parser.add_argument('--benchmark', '-bm', type=str, default='protein_structure', help='benchamrk name')
    parser.add_argument('--budget', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--suffix', '-suf', type=str, default='', help='experiment suffix')
    parser.add_argument('--max_iter', '-mi', default=12, type=int,
                        help='number of expensive iters, including the initial points')
    parser.add_argument('--ninit', '-ni', default=3, type=int, help='number of init')
    parser.add_argument('--nseed', '-ns', default=1, type=int, help='number of random seeds')
    parser.add_argument('--root', '-r', default='runs_jumbo', type=str,
                        help='The root directory for book keeping')
    parser.add_argument('--gp_warm_train_max_epoch', '-gpw_n', default=100, type=int,
                        help='Number of Warm GP training epochs per iteration')
    parser.add_argument('--gp_warm_hyper_lr', default=0.1, type=float,
                        help='Learning rate for kernel hyper parameters of the Warm GP')
    parser.add_argument('--gp_cold_train_max_epoch', '-gpc_n', default=100, type=int,
                        help='Number of Cold GP training epochs per iteration')
    parser.add_argument('--gp_cold_hyper_lr', default=0.1, type=float,
                        help='Learning rate for kernel hyper parameters of the Cold GP')
    parser.add_argument('--alpha_threshold', '-la', default=0.1, type=float,
                        help='The coefficient hyper parameter in the acquisition function, '
                             'd=inf means a vanilla BO.')
    parser.add_argument('--use_latent', action='store_true',
                        help='True to use latent instead of the output of aux model')
    return parser.parse_args()


if __name__ == '__main__':

    _args = _arg_parse()


    fn = get_eval_function(_args.benchmark, _args.budget)
    tr = get_transform(_args.benchmark)

    n_init = _args.ninit
    max_iter = _args.max_iter
    gp_warm_train_cf = dict(
        max_epochs=_args.gp_warm_train_max_epoch,
        learning_rate=_args.gp_warm_hyper_lr,
    )
    gp_cold_train_cf = dict(
        max_epochs=_args.gp_cold_train_max_epoch,
        learning_rate=_args.gp_cold_hyper_lr,
    )
    seeds = [i * 10 for i in range(_args.nseed)]

    for i in range(_args.nseed):
        seed = i * 10
        jumbo_optimize(
            objective=fn,
            n_init=n_init,
            num_bo_iter=max_iter,
            rng=seed,
            root=_args.root,
            pretrained_model_path=_args.pretrained_path,
            aux_output_indices=slice(-1, None),
            transform=tr,
            query_cost=False,
            gp_warm_train_cf=gp_warm_train_cf,
            gp_cold_train_cf=gp_cold_train_cf,
            alpha_threshold=_args.alpha_threshold,
            exp_suffix=_args.suffix,
            use_latent=_args.use_latent,
        )