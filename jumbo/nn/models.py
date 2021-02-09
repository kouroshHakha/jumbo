from typing import Union, Tuple, List, Callable

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam

from jumbo.nn.basics import (
    TrainerConfig, TrainerModel, ModelCfBase
)
from jumbo.util import read_yaml

@dataclass_json
@dataclass
class ModelCf(ModelCfBase):
    input_dim: int
    out_dim: int
    xstat: Tuple = None
    ystat: Tuple = None
    nhidden: int = 32
    nz_dim: int = 4
    optimize_out: bool = True
    nh_layer: int = 3
    activation: Union[str, List[str]] = 'tanh'
    output_act: str = 'tanh'
    skip_conn: bool = False
    bnorm: bool = False
    drop_out: float = 0
    use_resnet: bool = False
    use_huber: bool = False
    input_transform_fn: Callable = None

    @classmethod
    def str_to_act(cls, act_str: str) -> nn.Module:
        act_str = act_str.lower()

        if act_str == 'tanh':
            return nn.Tanh()
        elif act_str == 'relu':
            return nn.ReLU()
        elif act_str == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f'activation {act_str} is not recognized!')


class ResNet(nn.Module):

    def __init__(self, n1, n2, act):
        super().__init__()


        self.fc = nn.Linear(n1, n2)
        self.act = act
        self.agg = nn.Linear(n1 + n2, n2)

    def forward(self, x):
        feat = self.act(self.fc(x))
        return self.agg(torch.cat([x, feat], dim=-1))


class Model(TrainerModel):

    def __init__(self, model_config: ModelCf = None):
        super().__init__(model_config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nhidden = model_config.nhidden
        nz_dim = model_config.nz_dim
        act = model_config.activation
        nlayer = model_config.nh_layer
        input_dim = model_config.input_dim
        xstat = model_config.xstat
        self.ystat = model_config.ystat
        self.skip_conn = model_config.skip_conn
        use_resnet = model_config.use_resnet


        if xstat:
            self.xstat = [torch.tensor(stat).to(self.device).float() for stat in xstat]
        else:
            self.xstat = None


        if model_config.input_transform_fn:
            self.input_tf = model_config.input_transform_fn
        else:
            self.input_tf = None

        if isinstance(act, str):
            interm_acts = [ModelCf.str_to_act(act) for _ in range(nlayer)]
        else:
            if len(act) != nlayer:
                raise ValueError(f'Must be given {nlayer + 1} activations')
            interm_acts = [ModelCf.str_to_act(a) for a in act]

        layers = []
        hiddens = [input_dim] + [nhidden] * nlayer
        self.output_act = ModelCf.str_to_act(model_config.output_act)
        for n1, n2, act in zip(hiddens[:-1], hiddens[1:], interm_acts):
            if use_resnet:
                layers.append(ResNet(n1, n2, act))
            else:
                layers.append(nn.Linear(n1, n2))
                layers.append(act)

        if self.skip_conn:
            self.feature = nn.Linear(nhidden + input_dim, nz_dim)
        else:
            self.feature  = nn.Linear(nhidden, nz_dim)
        self.in_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(nz_dim, model_config.out_dim)
        self.drop_out = nn.Dropout(p=model_config.drop_out)

        if model_config.bnorm:
            self.bnorm = nn.BatchNorm1d(nhidden)
        else:
            self.bnorm = None

        self.apply(self.weight_init)


    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizer(self, config: TrainerConfig, **kwargs):
        return Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, x, return_latent: bool = False):

        if self.input_tf:
            x = self.input_tf(x)
        elif self.xstat:
            x_mean = self.xstat[0]
            x_std = self.xstat[1]
            x = (x - x_mean) / x_std

        out_in_layers = self.in_layers(x)

        if self.skip_conn:
            if self.bnorm:
                out_in_layers = self.bnorm(out_in_layers)
            out_in_layers = self.drop_out(out_in_layers)
            feature_layer_in = torch.cat([x, out_in_layers], dim=-1)
        else:
            feature_layer_in = self.drop_out(out_in_layers)

        feature_out = self.output_act(self.feature(feature_layer_in))

        out = self.out_layer(feature_out)
        if return_latent:
            return out, feature_out
        return out

    def get_loss(self, x, y, use_huber=True):

        pred, latent = self(x, return_latent=True)

        if y.ndim == 1:
            y = y[:, None]

        if pred.ndim == 1:
            pred = pred[:, None]

        y_rxpnd = y.unsqueeze(0).expand(y.shape[0], y.shape[0], -1)
        y_cxpnd = y_rxpnd.transpose(0, 1)

        # construct a 3D matrix that facilitates computing phi(i) - phi(j) for all i, j
        latent_rxpnd = latent.unsqueeze(0).expand(latent.shape[0], latent.shape[0], -1)
        latent_cxpnd = latent_rxpnd.transpose(0, 1)

        # the upper triangle of this matrix is what we want.
        inds = torch.triu_indices(latent.shape[0], latent.shape[0], 1)
        phi_ij = (latent_rxpnd - latent_cxpnd)[tuple(inds)]
        y_ij = (y_rxpnd - y_cxpnd)[tuple(inds)]

        phi_norm = (phi_ij ** 2).sum(-1)
        y_norm = (y_ij ** 2).sum(-1)

        use_huber = use_huber and self.config.use_huber
        if use_huber:
            latent_loss = nn.SmoothL1Loss()(phi_norm, y_norm)
        else:
            latent_loss = nn.MSELoss()(phi_norm, y_norm)

        if self.config.optimize_out:
            main_loss = nn.MSELoss()(pred, y)
            return latent_loss + main_loss
        return latent_loss

class SimpleModel(Model):
    """ This model only minimizes the output loss and nothing else"""
    def get_loss(self, x, y, use_huber=True):
        pred, _ = self(x, return_latent=True)
        if pred.ndim == 1:
            pred = pred.unsqueeze(-1)
        main_loss = nn.MSELoss()(pred, y)
        return main_loss


def get_model(path: str, pretrained: bool = True, model_mode: str = 'old'):

    configs = read_yaml(Path(path) / 'config.yaml')
    model_config = ModelCf(**configs)

    if model_mode == 'old':
        model: Model = Model(model_config)
    else:
        model: SimpleModel = SimpleModel(model_config)

    if pretrained:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(Path(path) / 'model.pt', map_location=device))

    return model