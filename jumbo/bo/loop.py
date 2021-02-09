from typing import Optional, Tuple

import torch
import time
from tqdm import tqdm
import numpy as np

from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints import Positive

from jumbo.nn.models import Model
from jumbo.acq.acq import (
    UpperConfidenceBoundWithAuxModel, MFAcquisition, UpperConfidenceBound
)
from jumbo.acq.acq_optimize import cma_minimize
from jumbo.bo.transform import (
    Transform, convert_to_xgrid_torch, tensor_x_to_tensor_grid
)
from jumbo.gp.model import GPCold, GPWarm
from jumbo.gp.train import GPTrainConfig, train


class Jumbo:

    def __init__(self,
                 objective_func,
                 lower,
                 upper,
                 aux_model: Model,
                 aux_output_indices = None,
                 transform: Optional[Transform] = None,
                 initial_points=3,
                 train_interval=1,
                 minimize=True,
                 rng=None,
                 ystats: Optional[Tuple[float, float]] = None,
                 train_cf1: GPTrainConfig = None,
                 train_cf2: GPTrainConfig = None,
                 alpha_threshold: float = 0.1,
                 use_latent=False,
                 ):

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng
        np.random.seed(20)

        self.X = None
        self.y = None
        self.ystats = ystats
        self.lower = lower
        self.upper = upper
        self.train_interval = train_interval
        self.objective_func = objective_func

        self.init_points = initial_points
        self.initial_design = init_latin_hypercube_sampling

        self.minimize = minimize

        self.T = None
        self.warning_cnts = 0
        self.acq_optim_fail = 0


        self.use_latent = use_latent
        self.aux_model: Model = aux_model.double()
        if aux_output_indices:
            self.slice = aux_output_indices
        else:
            self.slice = slice(-1, None)

        self.train_cf1 = train_cf1
        self.train_cf2 = train_cf2
        self.alpha_threshold = alpha_threshold

        self.transform = transform

        # assuming output is standard normalized set lower and upper to be +- 3 * sigma
        y1_stat = self.aux_model.ystat
        if use_latent:
            self.y1_lower = -torch.ones(len(y1_stat[1]))[..., self.slice]
            self.y1_upper = -self.y1_lower
        else:
            self.y1_lower = -3 * torch.ones(len(y1_stat[1]))[..., self.slice]
            self.y1_upper = -self.y1_lower


    def run(self, num_iterations=10, X=None, y=None, query_cost:bool = False):
        # num_iterations is the minimum number of bo iterations if max_time is not reached yet
        s = time.time()
        self.T = num_iterations

        runtime, time_overhead = [], []
        incumbents, incumbents_values = [], []

        running_cost = 0
        if X is None and y is None:
            # Initial design
            X = []
            y = []

            init = self.initial_design(self.lower, self.upper, self.init_points, rng=self.rng)

            pbar = tqdm(enumerate(init), total=len(init))
            for i, x in pbar:

                if query_cost:
                    new_y, new_c = self.objective_func(x, minimize=self.minimize, return_cost=True)
                    running_cost += float(new_c)
                else:
                    new_y = self.objective_func(x, minimize=self.minimize)

                X.append(x)
                y.append(float(new_y))


                # Use best point seen so far as incumbent
                best_idx = int(np.argmin(y))
                incumbent = X[best_idx]
                incumbent_value = y[best_idx]

                incumbents.append(incumbent.tolist())
                incumbents_values.append(incumbent_value)
                pbar.set_description(f'collecting init points, f_opt = {incumbent_value:.6f}')

                time_overhead.append(0)
                if query_cost:
                    runtime.append(running_cost)
                else:
                    e = time.time() - s
                    runtime.append(e)

            self.X = np.array(X)
            self.y = np.array(y)

        else:
            self.X = X
            self.y = y


        it = 0
        pbar = tqdm(range(num_iterations - self.init_points))
        while it in pbar:

            if it % self.train_interval == 0:
                do_optimize = True
            else:
                do_optimize = False

            # Choose next point to evaluate
            s_choose = time.time()
            new_x = self.choose_next(self.X, self.y, it, do_optimize=do_optimize)[0]
            time_overhead.append(time.time() - s_choose)

            # Evaluate
            if query_cost:
                new_y, new_c = self.objective_func(new_x, minimize=self.minimize, return_cost=True)
                running_cost += new_c
                cur_t = float(running_cost + sum(time_overhead))
                runtime.append(cur_t)
            else:
                new_y = self.objective_func(new_x, minimize=self.minimize)
                cur_t = time.time() - s
                runtime.append(cur_t)

            # Extend the data
            self.X = np.append(self.X, new_x[None, :], axis=0)
            self.y = np.append(self.y, float(new_y))

            # Estimate incumbent
            best_idx = np.argmin(self.y)
            incumbent = self.X[best_idx]
            incumbent_value = self.y[best_idx]

            incumbents.append(incumbent.tolist())
            incumbents_values.append(incumbent_value)

            msg = f'[iter {it + 1} / {num_iterations - self.init_points}] ' \
                  f'f_opt = {incumbent_value:.6f}, ' \
                  f'last_fval = {self.y[-1]:.6f}'
            pbar.set_description(msg)

            it += 1

        results = dict(
            x_opt = list(incumbents[-1]),
            f_opt = float(incumbents_values[-1]),
            incumbents = [list(inc) for inc in incumbents],
            incumbent_values = [float(val) for val in incumbents_values],
            runtime = runtime,
            overhead = time_overhead,
            X = [x.tolist() for x in self.X],
            y = [y.tolist() for y in self.y],
        )

        print(f'Total Acquisition failures: {self.warning_cnts}')
        print(f'Complete Acquisition failures: {self.acq_optim_fail}')

        return results

    def choose_next(self, X: np.ndarray, y: np.ndarray, iteration: int, do_optimize=True):
        y2_stat = y.mean(0), y.std(0)
        y2_standard = (y - y2_stat[0]) / (y2_stat[1] + 1e-12)

        # gp_warm: y1 -> y2
        # gp_cold: x -> y2
        self.aux_model.eval()
        warm_gp, cold_gp = self._train_gp_models(X, y2_standard)

        tau = self.T / np.log(10)
        beta = 10 * np.exp(-iteration / tau)

        acqf_warm = UpperConfidenceBoundWithAuxModel(warm_gp, aux_model=self.aux_model,
                                                     aux_bounds=[self.y1_lower, self.y1_upper],
                                                     beta=beta,
                                                     maximize=not self.minimize,
                                                     slice=self.slice,
                                                     use_latent=self.use_latent,
                                                     transform=self.transform)

        xstar_warm, warm_optim = self._optimize(acqf_warm)

        acqf_cold = UpperConfidenceBound(cold_gp, beta=beta, maximize=not self.minimize)

        acqf = MFAcquisition(cold_gp, acqf_warm, acqf_cold, warm_optim,
                             alpha_thresh=self.alpha_threshold)

        new_x, new_optim_value = self._optimize(acqf)

        return new_x.detach().cpu().numpy()


    def _train_gp_models(self, x, y2):
        X = torch.tensor(x)
        y2 = torch.tensor(y2)

        ll1 = GaussianLikelihood()
        ll2 = GaussianLikelihood(noise_constraint=Positive())

        Xgrid = convert_to_xgrid_torch(X, self.transform).double()
        y1_pred, y1_latent = self.aux_model(Xgrid, return_latent=True)
        train_y1 = y1_latent if self.use_latent else y1_pred
        train_y1 = (train_y1.data[..., self.slice] - self.y1_lower) / (self.y1_upper - self.y1_lower)
        warm_gp = GPWarm(train_y1, y2, ll1)
        train(train_y1, y2, warm_gp, self.train_cf1)

        transform_input_fn = tensor_x_to_tensor_grid(self.transform)
        cold_gp = GPCold(X, y2, ll2, transform_input_fn=transform_input_fn)
        train(X, y2, cold_gp, self.train_cf2)
        return warm_gp, cold_gp

    def _optimize(self,  acqf):
        acqf.model.eval()
        best_idx = np.argmin(self.y)

        optim_acq_val = acqf(torch.tensor([self.X[best_idx]]).unsqueeze(-2)).item()

        cnt, cnt_max = 0, 10
        new_optim_val, best_x, best_temp_acq = float('inf'), None, float('inf')
        while optim_acq_val < new_optim_val and cnt < cnt_max:
            x, new_optim_val = cma_minimize(acqf, bounds=np.stack([self.lower, self.upper]),
                                            maxiter=100, sigma0=0.5)
            if optim_acq_val < new_optim_val:
                print(f'[Warning] Acquisition function is not improved compared to '
                      f'acq_f = {optim_acq_val}, trying again ...')
            if new_optim_val <= best_temp_acq:
                best_x = x
                best_temp_acq = new_optim_val

            cnt += 1
            if cnt == cnt_max:
                print('[Warniing] Acquisition function optimization has become very hard, '
                      'using the best value so far + added noise ...')
                new_optim_val = best_temp_acq

        return best_x, new_optim_val


def init_latin_hypercube_sampling(lower, upper, n_points, rng=None):
    """
    Returns as initial design a N data points sampled from a latin hypercube.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bound of the input space
    upper: np.ndarray (D)
        Upper bound of the input space
    n_points: int
        The number of initial data points
    rng: np.random.RandomState
            Random number generator

    Returns
    -------
    np.ndarray(N,D)
        The initial design data points
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))
    n_dims = lower.shape[0]
    # Generate bounds for random number generator
    s_bounds = np.array([np.linspace(lower[i], upper[i], n_points + 1) for i in range(n_dims)])
    s_lower = s_bounds[:, :-1]
    s_upper = s_bounds[:, 1:]
    # Generate samples
    samples = s_lower + rng.uniform(0, 1, s_lower.shape) * (s_upper - s_lower)
    # Shuffle samples in each dimension
    for i in range(n_dims):
        rng.shuffle(samples[i, :])
    return samples.T