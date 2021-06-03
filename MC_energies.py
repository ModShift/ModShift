import numpy as np
import os
import yaml
import torch
from utils.dataloader import load_data
from mod_shift.ModShiftModel_keops import ModShiftModel
from mod_shift.implemented_functions import w_clipped_linear, rho_clipped_linear

datasets = ["CREMI_A", "CREMI_B", "CREMI_C", "ISBI"]
repeats = 5
downsample = 4

# get weights as LazyTensor
def compute_weights_from_points(data):
    dists, _, _ = ModShiftModel.compute_distances(data)
    return w_clipped_linear(dists, clustering_config[dataset+'_ModShift']['betas'][0])

# get rhos as LazyTensor
def compute_rho_from_points(data):

    dists, _, _ = ModShiftModel.compute_distances(data)

    return rho_clipped_linear(dists, 2 * clustering_config[dataset+'_ModShift']['betas'][0])

# compute MC vectors  from labels using pykeops
def compute_mc_vector_from_seg(seg):
    seg = torch.tensor(seg, device="cuda")[:, None, ...]

    dists, _, _ = ModShiftModel.compute_distances(seg)

    # mc vector in matrix form, so every non-diagonal entry appears twice
    mc_vector = dists.sign()  # 0 for dist = 0, 1 for dist > 0
    return mc_vector

# compute energies
def mc_energy(mc_vector, weights):
   return 0.5*(mc_vector * weights).sum(1).sum(1)  # divide by 2 as every edge is considered twice


for dataset in datasets:
    print(f"{dataset} #############################################")
    # load data config
    with open(f"config_{dataset}.yml", "r") as config_file:
        data_config = yaml.load(config_file, Loader=yaml.FullLoader)

    # load weights config
    with open(f"config_{dataset}_best.yml", "r") as config_file:
        clustering_config = yaml.load(config_file, Loader=yaml.FullLoader)

    # set target_dir
    target_dir = os.path.join(data_config["root_path"], data_config["dataset"], "results")
    if data_config["dataset"] =="CREMI":
        target_dir = os.path.join(target_dir, data_config["set"])

    # load gt
    data = load_data(data_config)[:, ::downsample, ::downsample, :]
    data = torch.tensor(np.moveaxis(data, -1, 1), device="cuda", dtype=torch.float64)
    gt = load_data(data_config, gt = True)[:, ::downsample, ::downsample].astype(np.float64)

    # load labels
    # load Mod Shift shifted points and labels
    mod_shifted = []
    mod_labels = []
    for i in range(repeats):
        mod_labels.append(np.load(os.path.join(target_dir,
                                               f"mod_shift_labels_beta_{clustering_config[dataset+'_ModShift']['betas'][0]}_"+ \
                                               f"threshold_{clustering_config[dataset+'_ModShift']['thresholds'][0]}_"+ \
                                               f"run_{i}_downsample_{downsample}.npy")))
        mod_shifted.append(np.load(os.path.join(target_dir,
                                               f"mod_shift_trajectories_beta_{clustering_config[dataset+'_ModShift']['betas'][0]}_"+ \
                                               f"run_{i}_downsample_{downsample}.npy"))[-1])
    mod_labels = np.stack(mod_labels, axis=0).astype(np.float64)  # runs slices H W
    mod_shifted = np.stack(mod_shifted, axis=0).astype(np.float64)
    mod_shifted = torch.tensor(mod_shifted, device="cuda", dtype=torch.float64)

    # load KL labels
    kl_labels = np.load(os.path.join(target_dir, f"KL_complete_labels_{repeats}.npy")).astype(np.float64)

    # load MWS labels
    mws_labels = np.load(os.path.join(target_dir, f"MWS_complete_labels_{repeats}.npy")).astype(np.float64)

    # compute energies
    weights = compute_weights_from_points(data)
    initial_energy = mc_energy(compute_rho_from_points(data), weights).mean(0)
    gt_energy = mc_energy(compute_mc_vector_from_seg(gt), weights).mean(0)
    mod_shifted_energy = []
    mod_energy = []
    kl_energy = []
    mws_energy = []
    for i in range(repeats):
        mod_shifted_energy.append(mc_energy(compute_rho_from_points(mod_shifted[i]), weights))
        mod_energy.append(mc_energy(compute_mc_vector_from_seg(mod_labels[i]), weights))
        kl_energy.append(mc_energy(compute_mc_vector_from_seg(kl_labels[i]), weights))
        mws_energy.append(mc_energy(compute_mc_vector_from_seg(mws_labels[i]), weights))

    mod_shifted_energy = torch.stack(mod_shifted_energy, dim=0).mean(1)
    mod_energy = torch.stack(mod_energy, dim=0).mean(1)
    kl_energy = torch.stack(kl_energy, dim=0).mean(1)
    mws_energy = torch.stack(mws_energy, dim=0).mean(1)

    mean_mod_shifted_energy = mod_shifted_energy.mean(0)
    std_mod_shifted_energy = mod_shifted_energy.std(0)

    mean_mod_energy = mod_energy.mean(0)
    std_mod_energy = mod_energy.std(0)

    mean_kl_energy = kl_energy.mean(0)
    std_kl_energy = kl_energy.std(0)

    mean_mws_energy = mws_energy.mean(0)
    std_mws_energy = mws_energy.std(0)

    # compare energies
    min_energy = min(initial_energy, mean_mod_energy, mean_kl_energy, mean_mws_energy, gt_energy, mean_mod_shifted_energy)

    deviation_inital = (initial_energy - min_energy)
    deviation_gt = (gt_energy - min_energy)
    deviation_mod_shifted = (mean_mod_shifted_energy - min_energy)
    deviation_mod = (mean_mod_energy - min_energy)
    deviation_kl = (mean_kl_energy - min_energy)
    deviation_mws = (mean_mws_energy - min_energy)

    # print
    print(f"Initial energy: {float(initial_energy)}")
    print(f"deviation initial energy: {float(deviation_inital)}\n")

    print(f"GT energy: {float(gt_energy)}")
    print(f"deviation gt energy: {float(deviation_gt)}\n")

    print(f"Mean Mod energy: {float(mean_mod_energy)}")
    print(f"Std Mod energy: {float(std_mod_energy)}")
    print(f"deviation Mod energy: {float(deviation_mod)}\n")

    print(f"Mean Mod shifted energy: {float(mean_mod_shifted_energy)}")
    print(f"Std Mod shifted energy: {float(std_mod_shifted_energy)}")
    print(f"deviation Mod shifted energy: {float(deviation_mod_shifted)}\n")

    print(f"Mean KL energy: {float(mean_kl_energy)}")
    print(f"Std KL energy: {float(std_kl_energy)}")
    print(f"deviation KL energy: {float(deviation_kl)}\n")

    print(f"Mean MWS energy: {float(mean_mws_energy)}")
    print(f"Std MWS energy: {float(std_mws_energy)}")
    print(f"deviation MWS energy {float(deviation_mws)}\n")

