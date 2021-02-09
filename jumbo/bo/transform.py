
import numpy as np
import ConfigSpace
import abc
import torch

class Transform(abc.ABC):
    # I don't think we'll need xgrid to anything

    def __init__(self, cs: ConfigSpace.ConfigurationSpace):
        self.cs = cs

    @abc.abstractmethod
    def transform_x_to_config(self, x: np.ndarray) -> ConfigSpace.Configuration:
        pass

    @abc.abstractmethod
    def transform_x_to_xgrid(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def transform_config_to_xgrid(self, config: ConfigSpace.Configuration) -> np.ndarray:
        pass


def convert_to_xgrid_torch(x: torch.Tensor, transform: Transform):
    if transform is None:
        return x

    batch_shape = x.shape[:-1]
    Xgrid = []
    for dsn in x.view(-1, x.shape[-1]):
        Xgrid.append(transform.transform_x_to_xgrid(dsn.detach().cpu().numpy()))
    Xgrid = torch.tensor(Xgrid)
    Xgrid = Xgrid.reshape(*batch_shape, Xgrid.shape[-1])
    return Xgrid

def convert_to_xgrid_np(x: np.ndarray, transform: Transform):
    if transform is None:
        return x

    batch_shape = x.shape[:-1]
    Xgrid = []
    for dsn in x.view(-1, x.shape[-1]):
        Xgrid.append(transform.transform_x_to_xgrid(dsn))
    Xgrid = np.array(Xgrid)
    Xgrid = Xgrid.reshape(*batch_shape, Xgrid.shape[-1])
    return Xgrid


def tensor_x_to_tensor_grid(transform_obj: Transform):

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        if transform_obj:
            batch_shape = x.shape[:-1]
            xtrans_list = []
            for xtens in x.view(-1, x.shape[-1]):
                xtrans_list.append(transform_obj.transform_x_to_xgrid(xtens.detach().cpu().numpy()))

            xtrans = torch.tensor(xtrans_list).to(x).reshape(*batch_shape, -1)
            return xtrans

        return x

    return transform_fn