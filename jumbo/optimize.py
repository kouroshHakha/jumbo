from typing import Union, Dict, Any, Optional

import numpy as np
import torch
from pathlib import Path

from jumbo.function_interface import EvalFunction
from jumbo.nn.models import get_model
from jumbo.gp.train import GPTrainConfig
from jumbo.util import write_yaml
from jumbo.bo.loop import Jumbo
from jumbo.bo.transform import Transform


def jumbo_optimize(objective: EvalFunction,
                   n_init: int,
                   num_bo_iter: int,
                   rng: Union[np.random.RandomState, int],
                   root='runs_jumbo',
                   pretrained_model_path='',
                   aux_output_indices=None,
                   transform: Optional[Transform] = None,
                   query_cost: bool = False,
                   gp_warm_train_cf: Dict[str, Any] = None,
                   gp_cold_train_cf: Dict[str, Any] = None,
                   alpha_threshold: float = 0.1,
                   exp_suffix: str = '',
                   use_latent: bool = False,
                   ):

    if isinstance(rng, int):
        seed = rng
        rng = np.random.RandomState(rng)
    else:
        seed = 0
    torch.manual_seed(seed)

    gp_warm_train_cf = {} or gp_warm_train_cf
    aux_model = get_model(pretrained_model_path, pretrained=True)

    train_cf1 = GPTrainConfig(**gp_warm_train_cf)
    train_cf2 = GPTrainConfig(**gp_cold_train_cf)

    gp_dir_name = f'{root}/jumbo_{objective.name}_d{objective.numDims}'
    if exp_suffix:
        gp_dir_name = f'{gp_dir_name}_{exp_suffix}'
    gp_dir_name = f'{gp_dir_name}/s{seed}'
    if Path(gp_dir_name).exists():
        print(f'path {gp_dir_name} already exists!')
        return

    jumbo = Jumbo(
        objective_func=objective,
        lower=objective.bounds[:, 0],
        upper=objective.bounds[:, 1],
        aux_model=aux_model,
        aux_output_indices=aux_output_indices,
        transform=transform,
        initial_points=n_init,
        train_cf1=train_cf1,
        train_cf2=train_cf2,
        rng=rng,
        ystats=None,
        alpha_threshold=alpha_threshold,
        use_latent=use_latent,
    )

    res = jumbo.run(num_iterations=num_bo_iter, query_cost=query_cost)

    write_yaml(f'{gp_dir_name}/res.yaml', res, mkdir=True)
