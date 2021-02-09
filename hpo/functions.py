from typing import Tuple

import os
import numpy as np

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

from jumbo.function_interface import EvalFunction

from hpo.transform import HPOTransform

TABULAR_ROOT = os.environ.get('NAS_ROOT', '~/Documents/dev/nas_benchmarks')

def get_eval_function(benchmark: str, budget: int = None) -> EvalFunction:

    b, min_budget, max_budget = get_tabular_bm_with_budget(benchmark)

    cs = b.get_configuration_space()
    tr = HPOTransform(cs)
    ndim = len(cs.get_hyperparameter_names())
    bounds = np.tile(np.array([0, 1]), (ndim, 1))

    if budget is None:
        budget = max_budget

    def fn(xarr: np.ndarray, return_cost=False, minimize=True):
        fvals, costs = [], []

        if xarr.ndim == 1:
            xarr = xarr[None, ...]

        for x in xarr:
            cf = tr.transform_x_to_config(x)

            if budget < min_budget or budget > max_budget:
                raise ValueError(f'budget is out of range [{min_budget}, {max_budget}]')
            fval, cost = b.objective_function(cf, budget=budget)

            if not minimize:
                fval = 1 - fval

            fvals.append(fval)
            costs.append(cost)

        if return_cost:
            return np.array(fvals), np.array(costs)
        return np.array(fvals)

    return EvalFunction(name=benchmark, function=fn, bounds=bounds)

def get_transform(benchmark: str) -> HPOTransform:
    bm, _, _ = get_tabular_bm_with_budget(benchmark)
    cs = bm.get_configuration_space()
    tr = HPOTransform(cs)
    return tr

def get_tabular_bm_with_budget(benchmark: str):
    nas_full_path = f'{TABULAR_ROOT}'
    fcnet_path = f'{TABULAR_ROOT}/fcnet_tabular_benchmarks'

    if benchmark == "nas_cifar10a":
        min_budget = 12
        max_budget = 108
        b = NASCifar10A(data_dir=nas_full_path)

    elif benchmark == "nas_cifar10b":
        b = NASCifar10B(data_dir=nas_full_path)
        min_budget = 12
        max_budget = 108

    elif benchmark == "nas_cifar10c":
        b = NASCifar10C(data_dir=nas_full_path)
        min_budget = 12
        max_budget = 108

    elif benchmark == "protein_structure":
        b = FCNetProteinStructureBenchmark(data_dir=fcnet_path)
        min_budget = 3
        max_budget = 100

    elif benchmark == "slice_localization":
        b = FCNetSliceLocalizationBenchmark(data_dir=fcnet_path)
        min_budget = 3
        max_budget = 100

    elif benchmark == "naval_propulsion":
        b = FCNetNavalPropulsionBenchmark(data_dir=fcnet_path)
        min_budget = 3
        max_budget = 100

    elif benchmark == "parkinsons_telemonitoring":
        b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=fcnet_path)
        min_budget = 3
        max_budget = 100
    else:
        raise ValueError(f'benchmark {benchmark} not recognized')

    return b, min_budget, max_budget