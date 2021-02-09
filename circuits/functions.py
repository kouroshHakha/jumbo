import numpy as np
import time

from bb_eval_engine.util.importlib import import_bb_env
from bb_eval_engine.base import EvaluationEngineBase
from bb_eval_engine.data.design import Design

from jumbo.function_interface import EvalFunction

from .utils import mapToOrigBounds, Discrete, Space


def get_eval_function(name: str) -> EvalFunction:

    if name == 'schematic':
        env = import_bb_env('bb_envs/src/bb_envs/ngspice/envs/two_stage_opamp_1.yaml')
    elif name == 'layout':
        env = import_bb_env('bb_envs/src/bb_envs/ngspice/envs/two_stage_opamp_4.yaml')
    else:
        raise ValueError(f'name {name} is not valid.')


    space_dict = {}
    for param_key, param_item in env.params.items():
        lo, hi, step = param_item
        space_dict[param_key] = Discrete(lo, hi, step)

    space = Space(space_dict)

    num_dim = 8
    bounds = np.tile(np.array([0, 1]), (num_dim, 1))


    def fn(x, minimize=True, return_cost=False):
        ys, rts = [], []
        # deal with batch of designs
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x[None, :]

        for xx in x:
            # deal with one dimensional inputs
            if xx.ndim == 1:
                xx = xx[None, :]

            np_bounds = np.array(env.input_bounds)
            xx = mapToOrigBounds(xx, np_bounds)
            xx = space.snap_to_grid(xx)
            dsn_objs = [Design(dsn.tolist(), env.spec_range.keys()) for dsn in xx]
            # we have to make sure designs have their value_dict setup before they are passed in
            for dsn in dsn_objs:
                for param_val, key in zip(dsn, env.params_vec.keys()):
                    dsn.value_dict[key] = param_val

            rt_s = time.time()
            dsn_objs = env.evaluate(dsn_objs, do_interpret=False)
            rt = time.time() - rt_s

            res = dsn_objs[0]['cost']
            y = res if minimize else -res
            if return_cost:
                ys.append(y)
                rts.append(rt)
            else:
                ys.append(y)

        if return_cost:
            return np.array(ys), np.array(rts)
        return np.array(ys)

    return EvalFunction(name=f'ckt_{name}', function=fn, bounds=bounds)