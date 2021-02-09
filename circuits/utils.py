
from typing import Mapping

import abc
from dataclasses import dataclass
import numpy as np
import scipy.special as special

@dataclass
class SpaceType(abc.ABC):

    @abc.abstractmethod
    def sample(self, n):
        raise NotImplementedError

    @abc.abstractmethod
    def low_bnd(self):
        raise NotImplementedError

    @abc.abstractmethod
    def high_bnd(self):
        raise NotImplementedError

    @abc.abstractmethod
    def snap_to_grid(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

@dataclass
class Continuous(SpaceType):
    lo: float
    hi: float

    def sample(self, n):
        return np.random.uniform(self.lo, self.hi, size=(n,1))

    def low_bnd(self):
        return np.array([self.lo])

    def high_bnd(self):
        return np.array([self.hi])

    def snap_to_grid(self, x: np.ndarray) -> np.ndarray:
        return x


@dataclass
class Discrete(SpaceType):
    # hi is inclusive
    lo: float
    hi: float
    step: float

    def __post_init__(self):
        self.vec = np.arange(self.lo, self.hi + self.step, self.step)

    def sample(self, n):
        return np.random.choice(self.vec, size=(n, 1), replace=True)

    def low_bnd(self):
        return np.array([self.vec[0]])

    def high_bnd(self):
        return np.array([self.vec[-1]])

    def snap_to_grid(self, x: np.ndarray) -> np.ndarray:
        # [lo1, lo1 + delta] -> lo1
        # (lo2 - delta , lo2 + delta] -> lo2
        # ...
        # (hi - delta, hi] -> hi
        half_vec = (self.vec[:-1] + self.vec[1:]) / 2.0
        grid = np.concatenate([np.array([-np.inf]), half_vec], axis=-1)
        bins = np.concatenate([np.array([-np.inf]), self.vec], axis=-1)
        x_inds = np.digitize(x, grid, right=False)
        x_copy = bins[x_inds]
        return x_copy

@dataclass
class Categorical(SpaceType):
    k: int
    one_hot: bool = True

    def sample(self, n):
        var = np.random.randint(self.k, size=(n, ))
        if not self.one_hot:
            return var.reshape(-1, 1)

        one_hot = np.zeros((var.shape[0], self.k))
        one_hot[np.arange(var.shape[0]), var] = 1
        return one_hot

    def low_bnd(self):
        if self.one_hot:
            return np.array([0.] * self.k)
        return np.array([0.])

    def high_bnd(self):
        if self.one_hot:
            return np.array([1.] * self.k)
        return np.array([self.k])


    def snap_to_grid(self, x: np.ndarray) -> np.ndarray:

        if self.one_hot:
            softmax = special.softmax(x, axis=-1)
            x_copy = np.zeros_like(x)
            x_copy[np.arange(x_copy.shape[0]), np.argmax(softmax, axis=-1)] = 1
            return x_copy

        vec = np.arange(self.k)
        half_vec = (vec[:-1] + vec[1:]) / 2.0
        grid = np.concatenate([np.array([-np.inf]), half_vec], axis=-1)
        bins = np.concatenate([np.array([-np.inf]), vec], axis=-1)
        x_inds = np.digitize(x, grid, right=False)
        x_copy = bins[x_inds]

        return x_copy

class Space:

    def __init__(self, space_info: Mapping[str, SpaceType]):
        self.space_info = space_info

        self.ndim: int = self._compute_ndim()
        self.bound: np.ndarray =  self._compute_bounds()


    def _compute_ndim(self):
        dim = 0
        for val in self.space_info.values():
            if isinstance(val, Categorical) and val.one_hot:
                dim += val.k
            else:
                dim += 1
        return dim


    def _compute_bounds(self):
        spaces = self.space_info.values()
        lo_bnd = np.concatenate([space.low_bnd() for space in spaces], axis=-1)
        hi_bnd = np.concatenate([space.high_bnd() for space in spaces], axis=-1)
        return np.stack([lo_bnd, hi_bnd], axis=0)


    def sample(self, n: int, return_dict: bool = False):

        samples = [space.sample(n) for space in self.space_info.values()]

        if return_dict:
            samples_dict_list = []
            for values in zip(*samples):
                sample_dict = {}
                for key, val in zip(self.space_info.keys(), values):
                    try:
                        sample_dict[key] = val.item()
                    except ValueError:
                        sample_dict[key] = val.tolist()
                samples_dict_list.append(sample_dict)
            return samples_dict_list

        return np.concatenate(samples, axis=-1)

    def snap_to_grid(self, x: np.ndarray):
        # x ~ N x D
        if x.ndim == 1:
            x = x[None, :]
        # TODO: maybe we want to support dictioanry input outputs and also conversion between array and dict from this object
        x_copy = np.clip(x, a_min=self.bound[0], a_max=self.bound[1])
        spaces = self.space_info.values()
        ptr = 0
        for space in spaces:
            offset = space.k if isinstance(space, Categorical) and space.one_hot else 1
            x_copy[:, ptr: ptr+offset] = space.snap_to_grid(x_copy[:, ptr: ptr+offset])
            ptr += offset
        return x_copy


def mapToOrigBounds(pt, bounds):
    return pt * (bounds[1] - bounds[0]) + bounds[0]