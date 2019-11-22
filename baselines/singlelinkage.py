import numpy as np
import os
import hdbscan
import sys
sys.path.append("..")
from utils.metrics import compute_metrics
from utils.obtain_hard_clusters import obtain_hard_clusters


class SingleLinkage(object):
    def __init__(self, config, target_dir):
        self.thresholds = config["thresholds"]
        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                print("Creation of the directory %s failed" % self.target_dir)

    def run(self, data, gt):
        labels = obtain_hard_clusters(data, self.thresholds)

        # compute scores, save labels and scores for each threshold
        results_list = []
        for i, threshold in enumerate(self.thresholds):
            np.save(os.path.join(self.target_dir, "single_linkage_labels_threshold_{}.npy".format(threshold)),
                    labels[:, i, ...])

            # compute metrics
            results = {"parameters": {"threshold": threshold},
                       "scores": compute_metrics(labels[:, i, ...], gt.copy())}
            # save results
            np.save(os.path.join(self.target_dir,
                                 "single_linkage_scores_threshold_{}".format(threshold)),
                    np.array([results["scores"]["CREMI_score"],
                              results["scores"]["arand"],
                              results["scores"]["voi"][0],
                              results["scores"]["voi"][1]
                              ]))
            results_list.append(results)

        return sorted(results_list, key=lambda x: x["scores"]["CREMI_score"])[0]




