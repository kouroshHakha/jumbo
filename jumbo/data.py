from typing import Any, Union, Dict

import time
import numpy as np
from pathlib import Path

from .function_interface import EvalFunction
from .util import read_pickle, write_pickle

def collect_data(rng: int,
                 npoints: int,
                 eval_f: EvalFunction,
                 query_cost: bool = False,
                 root: str = '.') -> Dict[str, Any]:

    root = Path(root) / 'data'
    fpath = root / f'{eval_f.name}_d{eval_f.numDims}_{npoints}_{rng}.pickle'

    if fpath.exists():
        print(f'file {str(fpath)} already exists!')
        return read_pickle(fpath)

    s = time.time()
    rng = np.random.RandomState(rng)

    x_tests = rng.rand(npoints, eval_f.numDims)
    y_tests = eval_f(x_tests, minimize=True)

    if query_cost:
        y_tests, cost_tests = eval_f(x_tests, return_cost=True, minimize=True)
        cost = sum(cost_tests)
    else:
        cost = time.time() - s

    ystat = (y_tests.mean(), y_tests.std())
    ynorm = (y_tests - ystat[0]) / ystat[1]
    xstat = (x_tests.mean(0), x_tests.std(0))

    data = dict(ynorm=ynorm, ystat=ystat, xarr=x_tests, xstat=xstat, cost=cost)
    write_pickle(fpath, data, mkdir=True)

    return data
