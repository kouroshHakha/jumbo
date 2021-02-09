from typing import Tuple

import numpy as np
import torch

try:
    from scipydirect import minimize
except ImportError:
    minimize = None

from cma import CMAEvolutionStrategy


def cma_minimize(acq_function, bounds, return_best_only=True, **kwargs) -> Tuple[torch.Tensor, ...]:
    x0 = 0.5 * np.ones(bounds.shape[-1])
    opts = {'bounds': [0, 1], "popsize": kwargs.get('popsize', 100), "seed": 10, "verbose": -1}
    if "maxiter" in kwargs:
        opts.update(maxiter=kwargs["maxiter"])
    es = CMAEvolutionStrategy(x0=x0, sigma0=kwargs.get('sigma0', 0.5), inopts=opts)

    xs_list, y_list = [], []
    with torch.no_grad():
        while not es.stop():
            xs = es.ask()
            X = torch.tensor(xs, dtype=torch.float64)
            Y = acq_function(X.unsqueeze(-2))
            y = Y.view(-1).double().numpy()
            es.tell(xs, y)
            xs_list.append(xs)
            y_list.append(y)

        if return_best_only:
            cand = torch.tensor([es.best.x])
            cand_val = torch.tensor([es.best.f])
        else:
            cand = torch.tensor(np.concatenate(xs_list, axis=0))
            cand_val = torch.tensor(np.concatenate(y_list, axis=0))

    return cand, cand_val

def direct_minimize(acq_function, bounds, return_best_only=True, **kwargs):
    def obj(x: np.ndarray):
        if x.ndim == 1:
            x = x[None, :]
        x = torch.tensor(x).double().unsqueeze(-2)
        y = acq_function(x)
        y = float(y.view(-1).item())

        return y

    res = minimize(obj, bounds.T)

    return torch.tensor(res.x).float().unsqueeze(0), torch.tensor(res.fun).float().unsqueeze(0)


def random_minimize(acq_function, bounds, return_best_only=True, **kwargs):
    xall = np.random.rand(10000, bounds.shape[-1]) * (bounds[1] - bounds[0]) + bounds[0]
    # xall = init_latin_hypercube_sampling(bounds[0], bounds[1], 50000)
    xall = torch.tensor(xall).double()
    yall = acq_function(xall.unsqueeze(-2))
    sorted_inds = yall.argsort()
    best_ind = sorted_inds[0]


    xbest = xall[best_ind].detach().cpu().unsqueeze(0)
    ybest = yall[best_ind].detach().cpu().unsqueeze(0)

    if return_best_only:
        return xbest, ybest

    return xall[sorted_inds], yall[sorted_inds]
