import hdbscan
import numpy as np
from tqdm import tqdm
import sys

def obtain_hard_clusters(points, thresholds):
    # expects points as B * ? * E, thresholds as list
    # returns single linkage labels of shape B len(thresholds) ?
    clusterer = hdbscan.HDBSCAN(min_samples=1, approx_min_span_tree=False)
    labels_list = []
    for batch_id in tqdm(range(points.shape[0]), desc="Processing batches", leave=False):
        clusterer.fit(points[batch_id].reshape(-1, points.shape[-1]))
        labels_batch = []
        for threshold in tqdm(thresholds, desc="Cutting at thresholds", leave=False):
            labels_batch_threshold = clusterer.single_linkage_tree_.get_clusters(cut_distance=threshold, min_cluster_size=1)
            labels_batch.append(labels_batch_threshold.reshape(*points.shape[1:-1]))
        labels_list.append(labels_batch)
        tqdm.write("Done with hard clustering batch {}".format(batch_id))
    return np.array(labels_list)

