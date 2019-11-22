from pykeops.torch import LazyTensor
from .implemented_functions import *
from torch import nn
import torch

import pykeops
pykeops.verbose = True

class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()

# computes the objective function of mod shift given distances between moving and initial points
class ModShiftModel(nn.Module):
    def __init__(self, config_rho, config_w, config):
        super().__init__()
        self.config = config
        self.config_rho = config_rho.copy()
        self.config_w = config_w.copy()

        # take care of relative / absolute d value
        self.config_rho["d"] = 2 * self.config_w["beta"]
        # create rho and weight function
        if self.config_rho["type"] == "clipped_linear":
            self.rho = rho_clipped_linear
        elif self.config_rho["type"] == "schoenberg":
            self.rho = rho_schoenberg
        else: print("Please specify valid rho.")

        if self.config_w["type"] == "clipped_linear":
            self.w = w_clipped_linear
        elif self.config_w["type"] == "clipped_cos":
            self.w = w_clipped_cos
        else: print("Please specify valid weight function")

    def forward(self, mv_points, points0=None):
        # set points0 to mv_points if no points0 are specified
        points0 = mv_points if points0 is None else points0
        # input dimensions are B(atch), E(mbd dim), ? (other dimensions)
        # pykeops expects embedding/channel dimension at the back and only one dimension for the points of one instance
        mv_points = mv_points.view(*(mv_points.size()[:2]), -1).permute(0, 2, 1).contiguous()
        mv_points_i = LazyTensor(mv_points[:, :, None, :])  # B N 1 E
        mv_points_j = LazyTensor(mv_points[:, None, :, :])  # B 1 N E
        diff_mv_points_ij = mv_points_i - mv_points_j  # B N N E
        dist_mv_points_ij = (diff_mv_points_ij**2).sum(-1).sqrt()  # B N N (1)
        if points0 is not None:
            points0 = points0.view(*(points0.size()[:2]), -1).permute(0, 2, 1).contiguous()
            points0_i = LazyTensor(points0[:, :, None, :])  # B N 1 E
            points0_j = LazyTensor(points0[:, None, :, :])  # B 1 N E
            diff_points0_ij = points0_i - points0_j  # B N N E
            dist_points0_ij = (diff_points0_ij**2).sum(-1).sqrt()  # B N N (1)
        else:
            dist_points0_ij = dist_mv_points_ij
        rho_ij = self.rho(dist_mv_points_ij, max_d=self.config_rho["d"])  # B N N (1)
        w_ij = self.w(dist_points0_ij, beta=self.config_w["beta"])  # B N N (1)
        loss_i = 0.5 * ((2 * rho_ij - 1) * w_ij).sum(2)  # B N (1) 1, last PyKeOps reduction step, output is torch.tensor
        loss_i = ContiguousBackward().apply(loss_i)  # backward pass needs to be contiguous as well
        loss = loss_i.sum(1)  # B (1, 1, 1) final reduction
        loss = loss.squeeze(0).squeeze(0)  # remove spurious dimension due to keops

        return 0.5*loss  # half the loss as all distances are considered twice
