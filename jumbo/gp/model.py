
from torch.optim import Adam

from botorch.models.gpytorch import GPyTorchModel

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP


class GPWarm(ExactGP, GPyTorchModel):

    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, outputscale=1.0):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroMean()
        self.kernel = ScaleKernel(MaternKernel(nu=2.5,
                                               # ard_num_dims=train_x.shape[-1]
                                               ))

        self.kernel.outputscale = outputscale
        # self.likelihood.noise_covar.noise = 0.2


    def configure_optimizer(self, train_cf):
        param_group = [
            {'params': self.kernel.base_kernel.parameters()},
            {'params': self.likelihood.noise_covar.parameters()},
        ]
        optim = Adam(param_group, lr=train_cf.learning_rate)
        return optim

    def forward(self, y1):
        mean = self.mean_module(y1)
        covar = self.kernel(y1)
        return MultivariateNormal(mean, covar)


class GPCold(ExactGP, GPyTorchModel):

    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, outputscale=10, transform_input_fn=None):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroMean()
        self.kernel = ScaleKernel(MaternKernel(nu=2.5))

        self.likelihood.noise_covar.noise = 1e-8
        self.kernel.outputscale = outputscale

        self.transform_input_fn = transform_input_fn

    def configure_optimizer(self, train_cf):
        param_group = [
            {'params': self.kernel.base_kernel.parameters()},
        ]
        optim = Adam(param_group, lr=train_cf.learning_rate)
        return optim

    def forward(self, x):
        if self.transform_input_fn:
            x = self.transform_input_fn(x)
        mean = self.mean_module(x)
        covar = self.kernel(x)
        return MultivariateNormal(mean, covar)

