import ConfigSpace
import numpy as np
import torch

from jumbo.bo.transform import Transform

class HPOTransform(Transform):

    def _check_x(self, x: np.ndarray):
        if len(x) != len(self.cs.get_hyperparameter_names()):
            raise ValueError('x should match the number of hyper parameters')

        if not np.all(np.bitwise_and(x >= 0, x <= 1)):
            raise ValueError('x should always be in range [0,1]')

    def transform_x_to_config(self, x: np.ndarray) -> ConfigSpace.Configuration:
        self._check_x(x)
        config_dict = {}
        for xi, h in zip(x, self.cs.get_hyperparameters()):
            if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                idx = self._snap_to_unit_cube_grid(xi, h.num_elements, False)
                config_dict[h.name] = h.get_value(idx)
            elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                cat_id = self._snap_to_unit_cube_grid(xi, len(h.choices), False)
                config_dict[h.name] = h.choices[cat_id]
            elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                config_dict[h.name] = int(xi * (h.upper - h.lower)) + h.lower
            elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                config_dict[h.name] = xi * (h.upper - h.lower) + h.lower

        return ConfigSpace.Configuration(self.cs, config_dict)


    def transform_x_to_xgrid(self, x: np.ndarray) -> np.ndarray:
        self._check_x(x)
        new_x = []
        for xi, h in zip(x, self.cs.get_hyperparameters()):
            if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                new_x.append(self._snap_to_unit_cube_grid(xi, h.num_elements, True))
            elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                cat_id = self._snap_to_unit_cube_grid(xi, len(h.choices), False)
                cat_onehot = list(map(int, list(format(1 << cat_id, f'0{len(h.choices)}b'))))
                new_x.extend(cat_onehot)
            elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                npoints = h.upper - h.lower
                new_x.append(int(xi * npoints) / npoints)
            elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                new_x.append(xi)
        return np.array(new_x)


    def transform_config_to_xgrid(self, config: ConfigSpace.Configuration) -> np.ndarray:
        new_x = []
        for h in self.cs.get_hyperparameters():
            value = config[h.name]
            if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                new_x.append(h.get_order(value) / h.num_elements)
            elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                cat_id = h.choices.index(value)
                cat_onehot = list(map(int, list(format(1 << cat_id, f'0{len(h.choices)}b'))))
                new_x.extend(cat_onehot)
            elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter or \
                    type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                new_x.append((value - h.lower) / (h.upper - h.lower))
        return np.array(new_x)

    def _snap_to_unit_cube_grid(self, x: float, np: int, divide: bool = False):
        idx = int(x * np)
        if idx == np:
            idx -= 1
        if divide:
            return idx / np
        return idx


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