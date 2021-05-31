from baselines.hdbscan import HDBSCAN
from baselines.meanshift_wrapper import MeanShiftWrapper
from baselines.completelinkage import CompleteLinkage
from baselines.singlelinkage import SingleLinkage
from mod_shift.ModShiftWrapper import ModShiftWrapper



def get_clusterer(method, config, target_dir, dataset=None):
    if method == "HDBSCAN":
        return HDBSCAN(config, target_dir)
    elif method == "MeanShift":
        return MeanShiftWrapper(config, dataset, target_dir)
    elif method == "ModShift":
        return ModShiftWrapper(config, dataset, target_dir)
    elif method == "CompleteLinkage":
        return CompleteLinkage(config, target_dir)
    elif method == "SingleLinkage":
        return SingleLinkage(config, target_dir)
    else:
        print("Invalid clustering method {}".format(method))



#