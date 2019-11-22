from torch import nn
from sklearn.neighbors import BallTree
from .implemented_functions import *
import torch
import numpy as np
from tqdm import tqdm


# computes the objective function of mod shift given distances between moving and initial points
class ModShiftModel(nn.Module):
    def __init__(self, config_rho, config_weight_func, config):
        super().__init__()
        self.config = config
        self.config_rho = config_rho.copy()
        self.config_weight_func = config_weight_func.copy()

        # take care of relative d value
        self.config_rho["d"] = 2 * self.config_weight_func["beta"]

        # create rho and weight function
        if config_rho["type"] == "clipped_parabola":
            self.rho = rho_clipped_parabola
        elif config_rho["type"] == "clipped_linear":
            self.rho = rho_clipped_linear
        elif config_rho["type"] == "clipped_cos":
            self.rho = rho_clipped_cos
        else: print("Please specify valid rho.")

        if config_weight_func["type"] == "clipped_linear":
            self.weight_func = w_clipped_linear
        elif config_weight_func["type"] == "clipped_cos":
            self.weight_func = w_clipped_cos
        else: print("Please specify valid weight function")

    def forward(self, dist_mv_points, dist_points0):
        return 0.5 * ((2 * self.rho(dist_mv_points, self.config_rho["d"]) - 1)
                         * self.weight_func(dist_points0, self.config_weight_func["beta"])).sum()


# computes the objective function for blocks of pairs of points, so that each computation fits on GPU
# can compute the gradients to the moving points directly, to avoid models that are too large
def compute_model(mv_points_batch, points0_batch, model, config):
    loss = 0
    num_points = np.array(mv_points_batch.shape[2:]).prod()
    # expects points as batchsize * channels (embd dim) * ?
    subsample_size = config["optimization"]["subsample_size"]
    view_idx = (mv_points_batch.shape[0], mv_points_batch.shape[1], -1)  # B E Others --> B E N

    # computes the contribution to the objective function of blocks of pairs of points in turn
    for i in tqdm(range(int(np.ceil(num_points/subsample_size))), desc="Subsample_i", leave=False):
        idx_i = i * subsample_size
        idx_i_max = np.minimum(idx_i + subsample_size, num_points)
        for j in tqdm(range(int(np.ceil(num_points/subsample_size))), desc="Subsample_j", leave=False):
            idx_j = j * subsample_size
            idx_j_max = np.minimum(idx_j + subsample_size, num_points)

            mv_points_subsample_i = mv_points_batch.view(view_idx)[:, :, idx_i:idx_i_max]
            mv_points_subsample_j = mv_points_batch.view(view_idx)[:, :, idx_j:idx_j_max]
            points0_subsample_i = points0_batch.view(view_idx)[:, :, idx_i:idx_i_max]
            points0_subsample_j = points0_batch.view(view_idx)[:, :, idx_j:idx_j_max]

            # obtain indices for upper triangular matrix, excluding the diagonal
            # compute distance of points, use every nontrivial distance once (upper triangle or dist matrices)
            dist_mv_points = torch.triu(torch.norm( mv_points_subsample_i.unsqueeze(3) -
                                                    mv_points_subsample_j.unsqueeze(2),
                                                    p=2, dim=1),
                                        diagonal=1)
            dist_points0 = torch.triu(torch.norm(points0_subsample_i.unsqueeze(3) -
                                                 points0_subsample_j.unsqueeze(2),
                                                 p=2, dim=1),
                                      diagonal=1)

            # already compute the gradients if the full model does not fit on GPU
            if config["optimization"]["compute_gradients_early"] == True:
                tmp_loss = model(dist_mv_points, dist_points0)
                loss += tmp_loss.detach()
                tmp_loss.backward()
                del tmp_loss
            else:
                loss += model(dist_mv_points, dist_points0)
            tqdm.write("Done with subsample {} of {}".format(i*int(np.ceil(num_points/subsample_size))+j, int(np.ceil(num_points/subsample_size))**2))
    return loss
