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
        self.beta = torch.tensor(self.config_w["beta"], device="cuda")
        self.grad_via_backprop = self.config["optimization"]["grad_via_backprop"]


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

        # create derivatives of rho and w (for both as we do not know if we do blurring or not) if grad not via backprop
        if not self.grad_via_backprop:
            if self.config_rho["type"] == "clipped_linear":
                self.derivative_rho = rho_clipped_linear_derivative
            elif self.config_rho["type"] == "schoenberg":
                self.derivative_rho = rho_schoenberg_derivative

            if self.config_w["type"] == "clipped_linear":
                self.derivative_w = w_clipped_linear_derivative
            elif self.config_w["type"] == "clipped_cos":
                self.derivative_w = w_clipped_cos_derivative
    # creates the maximal distance for rho on the fly as we cannot pickle lazy tensor attributes and hence avoid them
    def d(self, beta=None):
        return 2*beta

    def forward_old(self, mv_points, points0=None):
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


    @staticmethod
    def compute_distances(mv_points, points0=None):
        # input dimensions are B(atch), E(mbd dim), ? (other dimensions)
        # pykeops wants embedding/channel dimension at the back and only one dimension for the points of one instance

        mv_points = mv_points.view(*(mv_points.size()[:2]), -1).permute(0, 2, 1).contiguous()
        mv_points_i = LazyTensor(mv_points[:, :, None, :])  # B N 1 E
        mv_points_j = LazyTensor(mv_points[:, None, :, :])  # B 1 N E
        diff_mv_points_ij = (mv_points_i - mv_points_j) #.abs()  # B N N E
        dist_mv_points_ij = (diff_mv_points_ij ** 2).sum(-1).sqrt()  # B N N (1)
        if points0 is not None:
            points0 = points0.view(*(points0.size()[:2]), -1).permute(0, 2, 1).contiguous()
            points0_i = LazyTensor(points0[:, :, None, :])  # B N 1 E
            points0_j = LazyTensor(points0[:, None, :, :])  # B 1 N E
            diff_points0_ij = points0_i - points0_j  # B N N E
            dist_points0_ij = (diff_points0_ij ** 2).sum(-1).sqrt()  # B N N (1)
        else:
            dist_points0_ij = dist_mv_points_ij

        return dist_mv_points_ij, dist_points0_ij, diff_mv_points_ij

    def forward(self, mv_points, points0=None):
        # input dimensions are B(atch), E(mbd dim), ? (other dimensions)
        data_size = mv_points.size()
        dist_mv_points_ij, dist_points0_ij, diff_mv_points_ij = self.compute_distances(mv_points, points0)

        lazy_beta = LazyTensor(self.beta)
        if self.grad_via_backprop:
            rho_ij = self.rho(dist_mv_points_ij, max_d=self.d(lazy_beta))  # B N N (1)
            w_ij = self.w(dist_points0_ij, beta=lazy_beta)  # B N N (1)
            loss_i = 0.5 * ((2 * rho_ij - 1) * w_ij).sum(2)  # B N (1) 1, last PyKeOps reduction step, output is torch.tensor
            loss_i = ContiguousBackward().apply(loss_i)  # backward pass needs to be contiguous as well
            loss = loss_i.sum(1)  # B (1, 1, 1) final reduction
            loss = loss.squeeze(0).squeeze(0).sum()  # remove spurious dimension due to keops
            return 0.5 * loss  # half the loss as all distances are considered twice
        else:
            w_ij = self.w(dist_points0_ij, beta=lazy_beta)  # B N N (1)
            derivative_rho_ij = self.derivative_rho(dist_mv_points_ij, max_d=self.d(lazy_beta))  # B N N (1)

            if points0 is not None:  # non-blurring case
                gradients = (derivative_rho_ij * w_ij *
                             diff_mv_points_ij.normalize()).sum(2)  # B N E   (1) # consider dividing diff by dist instead of normalising as dist is already computed
            else:  # blurring case
                rho_ij = self.rho(dist_mv_points_ij, max_d=self.d(lazy_beta))  # B N N (1)
                derivative_w_ij = self.derivative_w(dist_points0_ij, beta=lazy_beta)  # B N N (1)

                gradients = ((derivative_rho_ij * w_ij + 0.5 * (2 * rho_ij - 1) * derivative_w_ij) *
                             diff_mv_points_ij.normalize()).sum(2)  # B N E(1), normalize operate per default on the last dim

            gradients = ContiguousBackward.apply(gradients).permute(0, 2, 1)  # B E N
            return gradients.view(data_size)  # B E ?
