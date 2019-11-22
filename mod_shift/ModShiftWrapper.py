import sys
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

    def run(self, data, gt):
        results_list = []
        for beta in tqdm(self.betas, desc="Processing beta", leave=False):
            self.config["weight_func"]["beta"] = beta
            ModShifter = ModShift(data, self.config)

            ModShifter.run(self.n_iter)
            trajectories = ModShifter.trajectories


            np.save(os.path.join(self.target_dir, "mod_shift_trajectories_beta_{}.npy".format(beta)), trajectories)

            labels = obtain_hard_clusters(np.moveaxis(trajectories[-1], 1, -1),
                                          [threshold if not (threshold == "same") else beta for threshold in self.thresholds])

            # compute scores, save labels and scores for each threshold
            for i, threshold in enumerate(self.thresholds):
                np.save(os.path.join(self.target_dir, "mod_shit_labels_beta_{}_threshold_{}.npy".format(beta, threshold)),
                        labels[:, i, ...])

                # compute metrics
                results = {"parameters": {"beta": beta, "threshold": threshold},
                           "scores": compute_metrics(labels[:, i, ...], gt.copy())}
                # save results
                np.save(os.path.join(self.target_dir,
                                     "mod_shift_scores_beta_{}_threshold_{}".format(beta, threshold)),
                        np.array([results["scores"]["CREMI_score"],
                                  results["scores"]["arand"],
                                  results["scores"]["voi"][0],
                                  results["scores"]["voi"][1]
                                  ]))
                results_list.append(results)
            tqdm.write("Done with beta {}".format(beta))
        return sorted(results_list, key=lambda x: x["scores"]["CREMI_score"])[0]



