import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm
from mod_shift.ModShift import ModShift
from baselines.meanshift import MeanShift
import yaml
import hdbscan
import torch
import matplotlib.pyplot as plt


# load config
with open("config_comparison_uniform.yml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

# instantiate hard clusterer
hard_clusterer = hdbscan.HDBSCAN(min_samples=1,
                                 approx_min_span_tree=False)


# run Mod Shift and Mean Shift and obtain the number of resulting clusters
num_clusters_mod_list =[]
num_clusters_ms_list = []
for i in tqdm(range(config["num_runs"]), desc="Runs", leave=False):
    np.random.seed(i)
    data = np.random.uniform(0, 1, 100).reshape(1, 100, 1)

    mod_shifter = ModShift(data, config["mod_shift"])
    mean_shifter = MeanShift(n_iter=config["mean_shift"]["epochs"],
                             bandwidth=config["mean_shift"]["bandwidth"],
                             kernel=config["mean_shift"]["kernel"],
                             blurring=config["mean_shift"]["blurring"],
                             use_keops=config["mean_shift"]["use_keops"]
                             )

    mod_shifter.run(config["mod_shift"]["optimization"]["num_epochs"])
    conv_points_mod = mod_shifter.trajectories[-1, 0].transpose()

    conv_points_mean = mean_shifter(torch.tensor(data, dtype=torch.float)).detach().cpu().numpy()[0]

    hard_clusterer.fit(conv_points_mod)
    labels_mod = hard_clusterer.single_linkage_tree_.get_clusters(cut_distance=config["threshold"],
                                                                  min_cluster_size=1)

    hard_clusterer.fit(conv_points_mean)
    labels_mean = hard_clusterer.single_linkage_tree_.get_clusters(cut_distance=config["threshold"],
                                                                   min_cluster_size=1)

    num_clusters_mod_list.append(labels_mod.max()+1)
    num_clusters_ms_list.append(labels_mean.max()+1)
    tqdm.write("Done with sample {}".format(i))

num_clusters_mod = np.array(num_clusters_mod_list)
num_clusters_ms = np.array(num_clusters_ms_list)

# save data
np.save("num_clusters_mod.npy", num_clusters_mod)
np.save("num_clusters_ms.npy", num_clusters_ms)

# print key statistics
print("Mean Mod: {}".format(np.mean(num_clusters_mod)))
print("Variance Mod: {}".format(np.var(num_clusters_mod)))
print("Mean Mean Shift: {}".format(np.mean(num_clusters_ms)))
print("Variance Mean Shift: {}".format(np.var(num_clusters_ms)))


# plot figure and save
max_num_clusters = np.maximum(num_clusters_mod.max(), num_clusters_ms.max())

plt.figure()
plt.hist(num_clusters_mod, bins=range(1, max_num_clusters+2), align="left", label="Mod Shift")
plt.hist(num_clusters_ms, bins=range(1, max_num_clusters+2), align="left", alpha=0.5, label="Mean Shift")
plt.legend()
plt.xlabel("Number of clusters")
plt.ylabel("Frequency")
plt.xticks(np.arange(1, 10,1).astype(int))


var_ms = np.var(num_clusters_ms)

plt.savefig("freq_ms_mod.png", dpi=300, bbox_inches="tight", pad_inches=0)















