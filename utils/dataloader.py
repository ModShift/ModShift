import h5py
import numpy as np
import os

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


