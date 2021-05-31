import sys
import time
sys.path.append("..")
from .ModShift import ModShift
from tqdm import tqdm
import numpy as np
import os
from utils.obtain_hard_clusters import obtain_hard_clusters
from utils.metrics import compute_metrics


class ModShiftWrapper(object):
    def __init__(self, config, dataset, target_dir):

        if dataset == "CREMI":
            self.betas = config["CREMI"]["betas"]
            self.thresholds = config["CREMI"]["thresholds"]
        elif dataset == "ISBI":
            self.betas = config["ISBI"]["betas"]
            self.thresholds = config["ISBI"]["thresholds"]
        else:
            print("Invalid dataset {} provided.".format(dataset))

        self.config = config["ModShift"]
        self.n_iter = config["ModShift"]["optimization"]["iterations"]

        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                print("Creation of the directory %s failed" % self.target_dir)

    def run(self, data, gt, repeats=1, timed=False, downsample=1):
        if timed:
            assert len(self.betas) == 1 and len(self.thresholds) == 1
        times = []
        for repeat in tqdm(range(repeats), desc="Runs"):
            results_list = []
            for beta in tqdm(self.betas, desc="Processing beta", leave=False):
                self.config["weight_func"]["beta"] = beta
                ModShifter = ModShift(data, self.config)

                time_before_run = time.perf_counter()
                ModShifter.run(self.n_iter)
                time_after_run = time.perf_counter()
                trajectories = ModShifter.trajectories

                np.save(os.path.join(self.target_dir, f"mod_shift_trajectories_beta_{beta}_run_{repeat}.npy"),
                        trajectories)

                time_before_hard_cluster = time.perf_counter()
                labels = obtain_hard_clusters(np.moveaxis(trajectories[-1], 1, -1),
                                              [threshold if not (threshold == "same") else beta for threshold in self.thresholds])
                time_after_hard_cluster = time.perf_counter()

                total_time = time_after_run - time_before_run + time_after_hard_cluster - time_before_hard_cluster
                times.append(total_time)
                # compute scores, save labels and scores for each threshold
                for i, threshold in enumerate(self.thresholds):
                    np.save(os.path.join(self.target_dir, f"mod_shit_labels_beta_{beta}_threshold_{threshold}_run_{repeat}.npy"),
                            labels[:, i, ...])

                    # compute metrics
                    results = {"parameters": {"beta": beta, "threshold": threshold},
                               "scores": compute_metrics(labels[:, i, ...], gt.copy())}
                    # save results
                    np.save(os.path.join(self.target_dir,
                                         f"mod_shift_scores_beta_{beta}_threshold_{threshold}_run_{repeat}.npy"),
                            np.array([results["scores"]["CREMI_score"],
                                      results["scores"]["arand"],
                                      results["scores"]["voi"][0],
                                      results["scores"]["voi"][1]
                                      ]))
                    results_list.append(results)
                tqdm.write("Done with beta {}".format(beta))
        times = np.array(times)
        if timed:
            print(f"Mean time: {times.mean(0)})")
            print(f"Std dev time: {times.std(0)}")
            np.save(os.path.join(self.target_dir,
                                 f"mod_shift_times_beta_{self.betas[0]}_threshold_{self.thresholds[0]}_"+
                                 f"downsample_{downsample}_repeats_{repeats}.npy"),
                    times)

        return sorted(results_list, key=lambda x: x["scores"]["CREMI_score"])[0]



