import argparse
import sys
sys.path.append("..")
import yaml
from utils.dataloader import load_data
from utils.clusterers import get_clusterer
import os

datasets = ["CREMI_A", "CREMI_B", "CREMI_C", "ISBI"]

for dataset in datasets:
    print(dataset)
    # load data config
    with open(f"config_{dataset}.yml", "r") as config_file:
        data_config = yaml.load(config_file, Loader=yaml.FullLoader)

    # load data and gt
    data = load_data(data_config)
    gt = load_data(data_config, gt = True)

    # load clusterer config
    with open(f"config_{dataset}_best.yml", "r") as config_file:
        clustering_config = yaml.load(config_file, Loader=yaml.FullLoader)

    # instantiate clustering
    target_dir = os.path.join(data_config["root_path"], data_config["dataset"], "results")
    if data_config["dataset"] =="CREMI":
        target_dir = os.path.join(target_dir, data_config["set"])

    # Mod Shift
    print("ModShift")
    # put configs for dataset (+subset) in the right place, overwriting previous one
    clustering_config[data_config["dataset"]] = clustering_config[dataset+"_"+"ModShift"]

    mod_shifter = get_clusterer("ModShift",
                              clustering_config,
                              target_dir=target_dir,
                              dataset= data_config["dataset"])
    # perform clustering
    best_results = mod_shifter.run(data, gt, repeats=5, timed=True)

    print("Best results for ")
    for parameter in best_results["parameters"]:
        print("{}: {}".format(parameter, best_results["parameters"][parameter]))
    print(best_results["scores"])

    print("downsampled")
    # perform clustering
    best_results = mod_shifter.run(data[:, ::4, ::4], gt[:, ::4, ::4], repeats=5, timed=True, downsample=4)

    print("Best results for ")
    for parameter in best_results["parameters"]:
        print("{}: {}".format(parameter, best_results["parameters"][parameter]))
    print(best_results["scores"])


    # Mean Shift
    print("Mean Shift")
    # put configs for dataset (+subset) in the right place, overwriting previous one
    clustering_config[data_config["dataset"]] = clustering_config[dataset+"_"+"MeanShift"]
    mean_shifter = get_clusterer("MeanShift",
                              clustering_config,
                              target_dir=target_dir,
                              dataset= data_config["dataset"])

    # perform clustering
    best_results = mean_shifter.run(data, gt, repeats=5, timed=True)

    print("Best results for ")
    for parameter in best_results["parameters"]:
        print("{}: {}".format(parameter, best_results["parameters"][parameter]))
    print(best_results["scores"])

    # HDBSCAN
    print("HDBSCAN")
    # put configs for dataset (+subset) in the right place, overwriting previous one
    clustering_config["min_samples"] = clustering_config[dataset+"_"+"HDBSCAN"]["min_samples"]
    clustering_config["min_cluster_sizes"] = clustering_config[dataset+"_"+"HDBSCAN"]["min_cluster_sizes"]
    hdbscanner = get_clusterer("HDBSCAN",
                              clustering_config,
                              target_dir=target_dir,
                              dataset= data_config["dataset"])

    # perform clustering
    best_results = hdbscanner.run(data, gt, repeats=5, timed=True)

    print("Best results for ")
    for parameter in best_results["parameters"]:
        print("{}: {}".format(parameter, best_results["parameters"][parameter]))
    print(best_results["scores"])

##### print all results again:
file_names = {
    "CREMI_A": {
        "ModShift": "mod_shift_times_beta_0.11_threshold_0.1_downsample_1_repeats_5.npy",
        "ModShift4": "mod_shift_times_beta_0.11_threshold_0.1_downsample_4_repeats_5.npy",
        "MeanShift": "mean_shift_times_kernel_bandwidth_0.11_threshold_0.11_repeats_5.npy",
        "HDBSCAN": "hdbscan_times_min_sample_50_min_cluster_50_repeats_5.npy"
    },
    "CREMI_B": {
        "ModShift": "mod_shift_times_beta_0.1_threshold_0.02_downsample_1_repeats_5.npy",
        "ModShift4": "mod_shift_times_beta_0.1_threshold_0.02_downsample_4_repeats_5.npy",
        "MeanShift": "mean_shift_times_kernel_bandwidth_0.09_threshold_0.1_repeats_5.npy",
        "HDBSCAN": "hdbscan_times_min_sample_10_min_cluster_300_repeats_5.npy"
    },
    "CREMI_C": {
        "ModShift": "mod_shift_times_beta_0.1_threshold_0.04_downsample_1_repeats_5.npy",
        "ModShift4": "mod_shift_times_beta_0.1_threshold_0.04_downsample_4_repeats_5.npy",
        "MeanShift": "mean_shift_times_kernel_bandwidth_0.06_threshold_0.1_repeats_5.npy",
        "HDBSCAN": "hdbscan_times_min_sample_20_min_cluster_150_repeats_5.npy"
    },
    "ISBI": {
        "ModShift": "mod_shift_times_beta_0.7_threshold_0.2_downsample_1_repeats_5.npy",
        "ModShift4": "mod_shift_times_beta_0.7_threshold_0.2_downsample_4_repeats_5.npy",
        "MeanShift": "mean_shift_times_kernel_bandwidth_0.5_threshold_1.0_repeats_5.npy",
        "HDBSCAN": "hdbscan_times_min_sample_20_min_cluster_350_repeats_5.npy"
    }
}
root_path = "./data/"


for dataset in file_names.keys():
    print(f"\n{dataset}" )
    data_path = os.path.join(root_path, dataset.split("_")[0], "results")
    if dataset.split("_")[0] == "CREMI":
        data_path = os.path.join(data_path, dataset.split("_")[1])

    for method in file_names[dataset].keys():
        run_times = np.load(os.path.join(data_path, file_names[dataset][method]))
        print(f"Mean run time {method} {dataset}: {run_times.mean(0)}")
        print(f"Std dev run time {method} {dataset}: {run_times.std(0)}\n")


