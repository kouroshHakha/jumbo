
import torch

from botorch.acquisition.analytic import (
    UpperConfidenceBound, AnalyticAcquisitionFunction
)
from jumbo.bo.transform import convert_to_xgrid_torch

class UpperConfidenceBoundWithAuxModel(UpperConfidenceBound):

    def __init__(self, model, aux_model,  aux_bounds, beta, slice, use_latent, transform, maximize=False):
        super().__init__(model=model, beta=beta, maximize=maximize)
        self.aux_model = aux_model
        self.upper = aux_bounds[1]
        self.lower = aux_bounds[0]
        self.slice = slice
        self.use_latent = use_latent
        self.transform = transform

    def forward(self, X):
        self.aux_model.eval()
        Xgrid = convert_to_xgrid_torch(X, self.transform).double()

        pred, latent = self.aux_model(Xgrid, return_latent=True)
        y1 = latent if self.use_latent else pred
        y1_normalized = (y1[..., self.slice] - self.lower) / (self.upper - self.lower)
        return super().forward(y1_normalized)

class MFAcquisition(AnalyticAcquisitionFunction):

    def __init__(self, model, acqf_warm, acqf_cold, warm_optim, alpha_thresh=0.1):
        super().__init__(model=model)

        self.acq_warm = acqf_warm
        self.acq_cold = acqf_cold

        self.warm_optim = warm_optim
        self.alpha_thresh = alpha_thresh

    def forward(self, X):
        """
        if x is bad based on its y1 then its acq1(x) should not be better than the current optimum
        of acq1. Therefore optim distance will be large and only acq1 will be considered as its
        acq function. If x is good based on its y1 and very similar to the current optimum then
        acq1 will not tell us anything, and we should consider x for determining what to sample
        next. if for some reason acq1 is better than acq1_optim we still want to use acq1 for
        surrogate, that's why we use and absolute.
        """
        optim_distance = (self.acq_warm(X) - self.warm_optim).abs()
        # coeff = torch.exp(-optim_distance / self.lengthscale)
        coeff = (optim_distance < self.alpha_thresh).float()
        acq = coeff * self.acq_cold(X) + (torch.tensor(1.) - coeff) * self.acq_warm(X)

        return acq
