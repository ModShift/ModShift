import numpy as np
import os
import hdbscan
import sys
sys.path.append("..")
from sklearn.neighbors import NearestNeighbors
from utils.metrics import compute_metrics
from tqdm import tqdm
import time



class HDBSCAN(object):
    def __init__(self, config, target_dir):
        self.min_samples_list = config["min_samples"]
        self.min_clusters_size_list = config["min_cluster_sizes"]
        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                print("Creation of the directory %s failed" % self.target_dir)

    def run(self, data, gt, repeats=1, timed=False):
        if timed:
            assert len(self.min_samples_list) == 1 and len(self.min_clusters_size_list) == 1
        times = []
        for repeat in tqdm(range(repeats), desc="Runs"):
            results_list = []
            for min_samples_name in tqdm(self.min_samples_list, desc="Processing min_samples", leave=False):
                for min_cluster_size in tqdm(self.min_clusters_size_list, desc="Processing min_cluster_size", leave=False):
                    if min_samples_name == "same":
                        min_samples = min_cluster_size
                    else:
                        min_samples = min_samples_name

                    clusterer = hdbscan.HDBSCAN(min_samples=min_samples,
                                                min_cluster_size=min_cluster_size,
                                                approx_min_span_tree=False)
                    labels_list = []
                    time_before_run = time.perf_counter()
                    for batch_id in tqdm(range(data.shape[0]), desc="Processing batch", leave=False):
                        clusterer.fit(data[batch_id].reshape(-1, data.shape[-1]))
                        labels_list.append(self.assign_noise(data[batch_id].reshape(-1, data.shape[-1]),
                                                             clusterer.labels_).reshape(data.shape[1: -1]))
                        tqdm.write("Done with clustering batch {}".format(batch_id))
                    time_after_run = time.perf_counter()
                    total_time = time_after_run - time_before_run
                    times.append(total_time)

                    labels = np.array(labels_list)

                    self.save_labels(labels, min_cluster_size, min_samples)

                    # compute metrics
                    results = {"parameters": {"min_samples": min_samples, "min_cluster_size": min_cluster_size},
                               "scores": compute_metrics(labels, gt.copy())}
                    # save results
                    np.save(os.path.join(self.target_dir,
                                         "hdbscan_scores_min_samples_{}_min_cluster_size_{}".format(min_cluster_size, min_samples)),
                            np.array([results["scores"]["CREMI_score"],
                                      results["scores"]["arand"],
                                      results["scores"]["voi"][0],
                                      results["scores"]["voi"][1]
                                      ]))
                    results_list.append(results)
                    tqdm.write("Done with min_samples {} and min_cluster_size {}".format(min_samples, min_cluster_size))
        times = np.array(times)
        if timed:
            print(f"Mean time: {times.mean(0)})")
            print(f"Std dev time: {times.std(0)}")
            np.save(os.path.join(self.target_dir,
                                 f"hdbscan_times_min_sample_{self.min_samples_list[0]}_"+
                                 f"min_cluster_{self.min_clusters_size_list[0]}_repeats_{repeats}.npy"),
                    times)


        return sorted(results_list, key=lambda x: x["scores"]["CREMI_score"])[0]

    def save_labels(self, labels, min_cluster_size, min_samples):
        np.save(os.path.join(self.target_dir,
                             "hdbscan_labels_min_samples_{}_min_cluster_size_{}".format(min_cluster_size, min_samples)),
                labels)


    # assigns label of nearest non-noise neighbor to points with noise label -1
    @staticmethod
    def assign_noise(data, labels):
        NN = NearestNeighbors(n_neighbors=1)
        NN.fit(data[labels != -1])
        nbrs_idx = NN.kneighbors(data[labels == -1], return_distance=False).reshape(-1)
        labels[labels == -1] = labels[labels != -1][nbrs_idx]
        return labels




