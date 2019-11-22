import numpy as np
import os
from fastcluster import complete
from tqdm import tqdm
import sys
sys.path.append("..")
from utils.metrics import compute_metrics
from scipy.cluster.hierarchy import fcluster


class CompleteLinkage(object):
    def __init__(self, config, target_dir):
        self.thresholds = config["thresholds"]
        self.target_dir = target_dir
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                print("Creation of the directory %s failed" % self.target_dir)

    def run(self, data, gt):
        results_list = []
        # for each batch compute complete linkage tree once, cut it at various thresholds
        labels_list = []
        for batch_id in tqdm(range(data.shape[0]), desc="Processing batch", leave=False):
            labels_batch = []
            complete_linkage = complete(data[batch_id].reshape(-1, data.shape[-1]))
            for threshold in tqdm(self.thresholds, desc="Cutting at thresholds", leave=False):
                labels_threshold_slice = fcluster(complete_linkage, t=threshold, criterion="distance")
                labels_threshold_slice = labels_threshold_slice.reshape(*data.shape[1:-1])
                labels_batch.append(labels_threshold_slice)
            labels_list.append(labels_batch)
            tqdm.write("Done with clustering batch {}".format(batch_id))

        labels = np.array(labels_list)

        # compute scores, save labels and scores for each threshold
        for i, threshold in enumerate(self.thresholds):
            np.save(os.path.join(self.target_dir, "complete_linkage_labels_threshold_{}.npy".format(threshold)),
                    labels[:, i, ...])

            # compute metrics
            results = {"parameters": {"threshold": threshold},
                       "scores": compute_metrics(labels[:, i, ...], gt.copy())}
            # save results
            np.save(os.path.join(self.target_dir,
                                 "complete_linkage_scores_threshold_{}".format(threshold)),
                    np.array([results["scores"]["CREMI_score"],
                              results["scores"]["arand"],
                              results["scores"]["voi"][0],
                              results["scores"]["voi"][1]
                              ]))
            results_list.append(results)

        return sorted(results_list, key=lambda x: x["scores"]["CREMI_score"])[0]




