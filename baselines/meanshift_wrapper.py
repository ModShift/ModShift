import sys
sys.path.append("..")
from .meanshift import MeanShift
from tqdm import tqdm
import torch
import numpy as np
import os
from utils.obtain_hard_clusters import obtain_hard_clusters
from utils.metrics import compute_metrics
import time

class MeanShiftWrapper(object):
    def __init__(self, config, dataset, target_dir):

        if dataset == "CREMI":
            self.bandwidths = config["CREMI"]["bandwidths"]
            self.thresholds = config["CREMI"]["thresholds"]
        elif dataset == "ISBI":
            self.bandwidths = config["ISBI"]["bandwidths"]
            self.thresholds = config["ISBI"]["thresholds"]
        else:
            print("Invalid dataset {} provided.".format(dataset))

        self.kernel = config["kernel"]
        self.blurring = config["blurring"]
        self.n_iter = config["iterations"]
        self.keops = config["keops"]
        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                print("Creation of the directory %s failed" % self.target_dir)

    def run(self, data, gt, repeats=1, timed=False):
        if timed:
            assert len(self.bandwidths) == 1 and len(self.thresholds) == 1
        times = []
        for repeat in tqdm(range(repeats), desc="Runs"):
            results_list = []
            for bandwidth in tqdm(self.bandwidths, desc="Processing bandwidth", leave=False):
                MeanShifter = MeanShift(n_iter = self.n_iter,
                                      bandwidth= bandwidth,
                                      kernel=self.kernel,
                                      blurring=self.blurring,
                                      use_keops=self.keops)
                time_before_run = time.perf_counter()
                convergence_points = MeanShifter(torch.tensor(data.reshape(data.shape[0], -1, data.shape[-1]))).detach().cpu().numpy()
                time_after_run = time.perf_counter()

                convergence_points.reshape(*data.shape)

                np.save(os.path.join(self.target_dir, "mean_shift_conv_points_bandwidth_{}.npy".format(bandwidth)), convergence_points)

                tqdm.write("Obtaining hard clustering")
                time_before_hard_cluster = time.perf_counter()

                labels = obtain_hard_clusters(convergence_points,
                                                [threshold if not (threshold == "same") else bandwidth for threshold in self.thresholds])
                time_after_hard_cluster = time.perf_counter()
                total_time = time_after_run - time_before_run + time_after_hard_cluster - time_before_hard_cluster
                times.append(total_time)

                # compute scores, save labels and scores for each threshold
                for i, threshold in enumerate(self.thresholds):
                    np.save(os.path.join(self.target_dir, "mean_shift_labels_bandwidth_{}_threshold_{}.npy".format(bandwidth, threshold)),
                            labels[:, i, ...])

                    # compute metrics
                    results = {"parameters": {"bandwidth": bandwidth, "threshold": threshold},
                               "scores": compute_metrics(labels[:, i, ...], gt.copy())}
                    # save results
                    np.save(os.path.join(self.target_dir,
                                         "mean_shift_scores_bandwidth_{}_threshold_{}".format(bandwidth, threshold)),
                            np.array([results["scores"]["CREMI_score"],
                                      results["scores"]["arand"],
                                      results["scores"]["voi"][0],
                                      results["scores"]["voi"][1]
                                      ]))
                    results_list.append(results)
                tqdm.write("Done with bandwidth {}".format(bandwidth))
        times = np.array(times)
        if timed:
            print(f"Mean time: {times.mean(0)})")
            print(f"Std dev time: {times.std(0)}")
            np.save(os.path.join(self.target_dir,
                                 f"mean_shift_times_kernel_bandwidth_{self.bandwidths[0]}_"+
                                 f"threshold_{self.thresholds[0]}_repeats_{repeats}.npy"),
                    times)

        return sorted(results_list, key=lambda x: x["scores"]["CREMI_score"])[0]



