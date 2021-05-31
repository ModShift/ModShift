import sys
sys.path.append("..")
from tqdm import tqdm
import numpy as np
import os
import nifty
import nifty.graph.opt.multicut as nmc

from utils.metrics import compute_metrics
from utils.dataloader import get_offsets, load_weights
from affogato.segmentation import compute_mws_segmentation_from_signed_affinities, compute_mws_clustering
from concurrent import futures
from functools import partial
import ray

import time

def parallel_threads(worker, n_slices):

    with futures.ThreadPoolExecutor(max_workers=n_slices) as tp:
        tasks = [tp.submit(worker, slice_id) for slice_id in range(n_slices)]
        labels = [t.result() for t in tasks]
    labels = np.array(labels)
    return labels

class GraphClusterer(object):
    def __init__(self, weights_config, data_config, graph_method_config, target_dir):
        self.graph_type = graph_method_config["graph_type"]

        assert self.graph_type == "offset" or self.graph_type == "complete", \
            f"'graph_type must be 'offset' or 'complete', not {self.graph_type}"

        self.weights_config = weights_config
        self.target_dir = target_dir
        self.data_config = data_config
        self.graph_method_config = graph_method_config

        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                print("Creation of the directory %s failed" % self.target_dir)

        if self.graph_type == "offset":
            self.weights = load_weights(data_config=data_config,
                                        weights_config=self.weights_config,
                                        graph_method_config=graph_method_config)
            self.weights = np.moveaxis(self.weights, 1, 0) # move axis over sliced to first positions
            self.offsets = get_offsets()
            self.image_shape = self.weights.shape[2:]

        elif self.graph_type == "complete":
            self.uvIds, self.weights = load_weights(data_config=data_config,
                                                    weights_config=self.weights_config,
                                                    graph_method_config=graph_method_config)
            image_side_length = int(np.sqrt(np.sqrt(2 * self.weights.shape[1] + 0.25 ) + 0.5))
            self.image_shape = (image_side_length, image_side_length)

        self.n_slices = len(self.weights)

    def compute_on_single_slice(self, slice_id):
        raise NotImplementedError

    def parallel_threads_wrapper(self):
        raise NotImplementedError

    def parallel_rays(self):
        raise NotImplementedError


    def run(self, gt, repeats, save=True):
        labels = []
        runtimes = []
        for repeat in tqdm(range(repeats), desc="Run"):
            # add tiny noise to weights to break equal weights (many are =-1)
            self.noisy_weights = self.weights + np.random.normal(0.0, 0.01, size=self.weights.shape)

            time_before = time.perf_counter()
            if self.graph_method_config["pool_executor"] == "thread":
                labels.append(parallel_threads(self.compute_on_single_slice, len(self.weights)))
            elif self.graph_method_config["pool_executor"] == "process":
                labels.append(self.parallel_rays())
            elif self.graph_method_config["pool_executor"] == "sequential":
                labels_per_run = []
                for slice_id in range(len(self.noisy_weights)):
                    labels_per_run.append(self.compute_on_single_slice(slice_id))
                labels_per_run = np.array(labels_per_run)
                labels.append(labels_per_run)
            time_after = time.perf_counter()
            runtimes.append(time_after - time_before)
        labels = np.stack(labels, axis=0).astype("int")
        runtimes = np.array(runtimes)

        if save:
            if self.graph_method_config["timed"]:
                np.save(os.path.join(self.target_dir,
                                     f"runtimes_{self.graph_method_config['method']}_{self.graph_type}_"+
                                     f"{self.graph_method_config['pool_executor']}_{repeats}.npy"),
                        runtimes)

            np.save(os.path.join(self.target_dir,
                                 f"{self.graph_method_config['method']}_{self.graph_type}_labels_{repeats}.npy"),
                    labels)
            scores = []

            for repeat in range(len(labels)):
                score_dict = compute_metrics(labels[repeat], gt.copy())
                scores.append(np.array([score_dict["CREMI_score"],
                                        score_dict["arand"],
                                        score_dict["voi"][0],
                                        score_dict["voi"][1]
                                        ]))
            scores = np.stack(scores, axis=0)

            np.save(os.path.join(self.target_dir,
                                 f"{self.graph_method_config['method']}_{self.graph_type}_scores_{repeats}.npy"),
                    scores)

            mean_scores = np.mean(scores, axis=0)
            std_scores = np.std(scores, axis=0)

            print(f"Mean CREMI score of {repeats} runs: {mean_scores[0]}")
            print(f"Std dev CREMI score of {repeats} runs: {std_scores[0]}")

            if self.graph_method_config["timed"]:
                print(f"Mean runtime of {repeats} runs: {runtimes.mean()}")
                print(f"Std dev runtime of {repeats} runs: {runtimes.std()}")


class MWS(GraphClusterer):
    def __init__(self, *args, **kwargs):
        super(MWS, self).__init__(*args, **kwargs)

    def compute_on_single_slice(self, slice_id):
        if self.graph_type == "offset":
            return compute_mws_segmentation_from_signed_affinities(self.noisy_weights[:, slice_id], self.offsets)
        elif self.graph_type == "complete":
            weights_slice = self.noisy_weights[slice_id]
            mask = weights_slice >= 0
            return compute_mws_clustering(self.image_shape[0] * self.image_shape[1], # number of pixels
                                          self.uvIds[mask], # attractive edges
                                          self.uvIds[~mask], # repulsive edges
                                          weights_slice[mask], # attractive weights
                                          np.abs(weights_slice[~mask]) # absolute value of repulsive weights
                                          ).reshape(self.image_shape)
    #TODO delete if still not used
    def parallel_threads_wrapper(self):
        if self.graph_type == "offset":
            partial_worker = partial(compute_mws_offset,
                                     offsets=self.offsets
                                     )
        elif self.graph_type == "complete":
            partial_worker = partial(compute_mws_complete,
                                     uvids=self.uvIds,
                                     image_shape=self.image_shape)
        else:
            raise NotImplementedError

        return parallel_threads(partial_worker, self.noisy_weights)

    def parallel_rays(self):
        if self.graph_type == "complete":
            partial_worker = partial(compute_mws_complete_ray,
                                     image_shape=self.image_shape)
        elif self.graph_type == "offset":
            partial_worker = partial(compute_mws_offset_ray,
                                     offsets=self.offsets)
        else:
            raise NotImplementedError

        ray.init(num_cpus=len(self.weights))
        weights_ptr = ray.put(self.noisy_weights)

        if self.graph_type == "offset":
            @ray.remote
            def remote_worker(slice_id, _weights_ptr):
                return partial_worker(slice_id, _weights_ptr)

            tasks = [remote_worker.remote(slice_id, weights_ptr) for slice_id in range(len(self.weights))]

        elif self.graph_type == "complete":
            uvids_ptr = ray.put(self.uvIds)
            @ray.remote
            def remote_worker(slice_id, _weights_ptr, _uvids_ptr):
                return partial_worker(slice_id, _weights_ptr, _uvids_ptr)

            tasks = [remote_worker.remote(slice_id, weights_ptr, uvids_ptr) for slice_id in range(len(self.weights))]
        else:
            raise NotImplementedError

        labels = np.array(ray.get(tasks))
        ray.shutdown()
        return labels

# functions for thread parallelism #TODO delete if still not used
def compute_mws_offset(weights, offsets):
    return compute_mws_segmentation_from_signed_affinities(weights, offsets)

def compute_mws_complete(weights, uvids, image_shape):
    mask = weights >= 0
    return compute_mws_clustering(image_shape[0] * image_shape[1],
                                  uvids[mask],
                                  uvids[~mask],
                                  weights[mask],
                                  np.abs(weights[~mask])
                                  ).rehsape(image_shape)

# functions for process parallelism via ray
def compute_mws_offset_ray(slice_id, weights, offsets):
    return compute_mws_segmentation_from_signed_affinities(weights[slice_id], offsets)

def compute_mws_complete_ray(slice_id, weights, uvids, image_shape):
    weights_slice = weights[slice_id]
    mask = weights_slice >= 0
    return compute_mws_clustering(image_shape[0] * image_shape[1],
                                  uvids[mask],
                                  uvids[~mask],
                                  weights_slice[mask],
                                  np.abs(weights_slice[~mask])
                                  ).reshape(image_shape)

class KL(GraphClusterer):
    is_local_offset = np.array([True, True, False, False, False, False, False, False, False, False, False, False])
    def __init__(self, *args, **kwargs):
        super(KL,self).__init__(*args, **kwargs)

        if self.graph_type == "offset":
            self.image_shape = self.weights.shape[2:]
            self.graph = nifty.graph.undirectedLongRangeGridGraph(self.image_shape,
                                                                  offsets=self.offsets,
                                                                  is_local_offset=KL.is_local_offset)
            self.mc_graph = nifty.graph.undirectedGraph(self.graph.numberOfNodes)
            self.mc_graph.insertEdges(self.graph.uvIds())

        elif self.graph_type == "complete":
            image_side_length = int(np.sqrt(np.sqrt(2 * self.weights.shape[1] + 0.25 ) + 0.5))
            self.image_shape = (image_side_length, image_side_length)
            # build graph
            self.graph = nifty.graph.undirectedGraph(self.image_shape[0]*self.image_shape[1])
            self.graph.insertEdges(self.uvIds)

    def compute_on_single_slice(self, slice_id):
        if self.graph_type == "offset":
            weights_slice = self.noisy_weights[:, slice_id]
            # build graph
            edge_values = self.graph.edgeValues(np.rollaxis(weights_slice, 0, 3))
            # run multicut
            obj = nmc.multicutObjective(self.mc_graph, edge_values)
            solver = obj.kernighanLinFactory(warmStartGreedy=True).create(obj)
            node_labels = solver.optimize().reshape(self.image_shape)
            return node_labels

        elif self.graph_type == "complete":
            weights_slice = self.noisy_weights[slice_id]
            # run multicut
            obj = nmc.multicutObjective(self.graph, weights_slice)
            solver = obj.kernighanLinFactory(warmStartGreedy=True).create(obj)
            node_labels = solver.optimize().reshape(self.image_shape)
            return node_labels

    def parallel_rays(self):
        if self.graph_type == "offset":
            partial_worker = partial(compute_kl_offset_ray,
                                     image_shape=self.image_shape)
        elif self.graph_type == "complete":
            partial_worker = partial(compute_kl_complete_ray,
                                     image_shape=self.image_shape)
        else:
            raise NotImplementedError

        ray.init(num_cpus=len(self.weights))
        weights_ptr = ray.put(self.noisy_weights)
        graph_ptr = ray.put(self.graph)

        if self.graph_type == "offset":
            mc_graph_ptr = ray.put(self.mc_graph)
            @ray.remote
            def remote_worker(slice_id, _weights_ptr, _graph_ptr, _mc_graph_ptr):
                return partial_worker(slice_id, _weights_ptr, _graph_ptr, _mc_graph_ptr)

            tasks = [remote_worker.remote(slice_id,
                                          weights_ptr,
                                          graph_ptr,
                                          mc_graph_ptr) for slice_id in range(len(self.weights))]

        elif self.graph_type == "complete":
            @ray.remote
            def remote_worker(slice_id, _weights_ptr, _graph_ptr):
                return partial_worker(slice_id, _weights_ptr, _graph_ptr)

            tasks = [remote_worker.remote(slice_id, weights_ptr, graph_ptr) for slice_id in range(len(self.weights))]
        else:
            raise NotImplementedError

        labels = np.array(ray.get(tasks))
        ray.shutdown()
        return labels


def compute_kl_offset_ray(slice_id, weights, graph, mc_graph, image_shape):
    weights_slice = weights[slice_id]
    # build graph
    edge_values = graph.edgeValues(np.rollaxis(weights_slice, 0, 3))
    # run multicut
    obj = nmc.multicutObjective(mc_graph, edge_values)
    solver = obj.kernighanLinFactory(warmStartGreedy=True).create(obj)
    node_labels = solver.optimize().reshape(image_shape)
    return node_labels

def compute_kl_complete_ray(slice_id, weights, graph, image_shape):
    weights_slice = weights[slice_id]
    # run multicut
    obj = nmc.multicutObjective(graph, weights_slice)
    solver = obj.kernighanLinFactory(warmStartGreedy=True).create(obj)
    node_labels = solver.optimize().reshape(image_shape)
    return node_labels




