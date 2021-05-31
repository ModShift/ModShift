import h5py
import numpy as np
import os
import torch
from mod_shift.implemented_functions import w_clipped_linear


# loads datasets in the format Batch * spatial dims * embedding dims


# individual functions for loading dataset
def load_cremi(dir_path, downsample_factor, slices, set, gt=False):
    file = h5py.File(os.path.join(dir_path, "CREMI", "data", "CREMI.h5"), "r") # E B H W
    if gt:
        if slices == "all":
            return file[set]["gt_seg"][:, ::downsample_factor, ::downsample_factor]  # B H W
        elif isinstance(slices, int):
            return file[set]["gt_seg"][slices, ::downsample_factor, ::downsample_factor][None, ...]  # 1 H W
        else:
            print("Invalid slice parameter {}".format(slice))

    else:
        if slices == "all":
            return file[set]["pred"][:, :, ::downsample_factor, ::downsample_factor].transpose(1, 2, 3, 0) # B H W E
        elif isinstance(slices, int):
            return file[set]["pred"][:, slices, ::downsample_factor, ::downsample_factor].transpose(1,2,0)[None, ...]  # 1 H W E
        else: print("Invalid slice parameter {}".format(slices))


def load_isbi(dir_path, downsample_factor, slices, gt=False):
    if gt:
        file = h5py.File(os.path.join(dir_path,"ISBI", "data", "ISBI.h5"), "r")
        if slices == "all":
            return file["gt_seg"][:, ::downsample_factor, ::downsample_factor] # B H W
        elif isinstance((slices, int)):
            return file["gt_seg"][slices, :: downsample_factor, ::downsample_factor][None, ...]  # 1 H W
        else:
            print("Invalid slice parameter {}".format(slices))
    else:
        data = np.load(os.path.join(dir_path, "ISBI", "data", "ISBI_embeddings_PCA_8.npy"))  # B E H W
        if slices == "all":
            return data[:, :, :: downsample_factor, ::downsample_factor].transpose((0, 2, 3, 1))  # B H W E
        elif isinstance((slices, int)):
            return data[slices, :, :: downsample_factor, ::downsample_factor].transpose((1, 2, 0))[None, ...]  # 1 H W E
        else:
            print("Invalid slice parameter {}".format(slices))


# abstracted dataset loading function
def load_data(data_config, gt=False):
    dataset = data_config["dataset"]

    if dataset == "CREMI":
        data = load_cremi(data_config["root_path"],
                          data_config["downsample_factor"],
                          data_config["slices"],
                          data_config["set"],
                          gt=gt)
    elif dataset == "ISBI":
        data = load_isbi(data_config["root_path"],
                         data_config["downsample_factor"],
                         data_config["slices"],
                         gt=gt)
    else: print("Please specify an implemented dataset.")
    return data

def get_offsets():
    return np.array([[-1, 0], [0, -1],
                    # indirect 3d nhood for dam edges
                    [-9, 0], [0, -9],
                    # long range direct hood
                    [-9, -9], [9, -9], [-9, -4], [-4, -9], [4, -9], [9, -4],
                    # inplane diagonal dam edges
                    [-27, 0], [0, -27]])

def compute_offset_weights_per_slice(points, get_weights, offsets):
    weights_list = []
    shape = points.shape[:2]
    for offset in offsets:
        assert offset[1] <= 0, "Offsets have incorrect signs"
        if np.abs(offset[1]) > shape[1] or np.abs(offset[0]) > shape[0]:
            print(f"Offset {offset} exceeded image dimensions, setting dummy weights of 0")
            weights = torch.zeros(shape)
        else:
            if offset[0] <= 0:
                dists = torch.norm(points[:shape[0] + offset[0], :shape[1] + offset[1]]
                                   - points[-offset[0]:, -offset[1]:],
                                   dim=-1,
                                   p=2)
                weights = get_weights(dists)
                weights = torch.nn.functional.pad(weights,
                                                  mode="constant",
                                                  value=0,
                                                  pad=(np.abs(offset[1]), 0, np.abs(offset[0]), 0))
            else:
                dists = torch.norm(points[:shape[0] - offset[0], -offset[1]:]
                                   - points[offset[0]:, :shape[1] + offset[1]],
                                   dim=-1,
                                   p=2)
                weights = get_weights(dists)
                weights = torch.nn.functional.pad(weights,
                                                  mode="constant",
                                                  value=0,
                                                  pad=(np.abs(offset[1]), 0, 0, np.abs(offset[0])))
        weights_list.append(weights)
    return torch.stack(weights_list)


def compute_complete_weights_per_slice(points, get_weights):
    points = points.reshape(-1, points.shape[-1])
    dists = torch.norm(points[:, None, :] - points[None, :, :], p=2, dim=-1)
    weights = get_weights(dists)

    idx = torch.triu_indices(points.shape[0], points.shape[0], offset=1)
    weights = weights[idx[0], idx[1]]


    assert len(weights) == points.shape[0] * (points.shape[0] - 1) / 2
    return weights, idx

def precompute_weights(data_config, weight_config, file_name, graph_method_config):
    print("Precomputing graph weights...")
    # get points
    points = torch.tensor(load_data(data_config))
    downsample_factor = graph_method_config["downsample_factor"]
    points = points[:, ::downsample_factor, ::downsample_factor, :]

    def get_weights(distance):
        return w_clipped_linear(distance, beta=weight_config["ModShift"]["weight_func"]["beta"])

    # get weights
    weights_by_slice = []

    if graph_method_config["graph_type"] == "offset":
        offsets = get_offsets()
        for z in range(len(points)):
            weights_by_slice.append(compute_offset_weights_per_slice(points[z], get_weights, offsets))

        weights = torch.stack(weights_by_slice, axis=1).numpy()
        with h5py.File(file_name, "w") as f:
            f.create_dataset("weights", data=weights)
        return_vals = weights

    elif graph_method_config["graph_type"] == "complete":
        uvIds = None
        for z in range(len(points)):
            weights_of_slice, idx = compute_complete_weights_per_slice(points[z], get_weights)
            weights_by_slice.append(weights_of_slice)
            if uvIds is None:
                uvIds = idx.T.numpy()
        weights = torch.stack(weights_by_slice, dim=0).numpy()

        with h5py.File(file_name, "w") as f:
            f.create_dataset("uvIds", data=uvIds)
            f.create_dataset("weights", dtype="float64", data=weights)
        return_vals =  uvIds, weights
    print("... done")
    return return_vals


def load_weights(weights_config, data_config, graph_method_config):
    if data_config["dataset"] == "CREMI":
        file_name = f"{graph_method_config['graph_type']}_weights_"+ \
                    f"{data_config['dataset']}{data_config['set']}_"+ \
                    f"downsample_{graph_method_config['downsample_factor']}.h5"
    else:
        file_name = f"{graph_method_config['graph_type']}_weights_" + \
                    f"{data_config['dataset']}_downsample_{graph_method_config['downsample_factor']}.h5"


    file_name = os.path.join(data_config["root_path"],
                             data_config["dataset"],
                             "data",
                             file_name)

    try:
        if graph_method_config["graph_type"] == "offset":
            with h5py.File(file_name, "r") as file:
                weights = np.array(file["weights"])
            return weights
        elif graph_method_config["graph_type"] == "complete":
            with h5py.File(file_name, "r") as file:
                uvIds = np.array(file["uvIds"])  # edges for a single complete graph
                weights = np.array(file["weights"])
            return uvIds, weights
    except:
        return precompute_weights(data_config, weights_config, file_name, graph_method_config)



