import argparse
import sys
sys.path.append("..")
import yaml
from baselines.graph_clusterer import MWS, KL
from utils.dataloader import load_data
import os

parser = argparse.ArgumentParser(description='Reproduce results of Mod Shift.')
parser.add_argument('--data', '-d',
                    help="""
                            The file specifying the dataset. One of 
                            config_CREMI_A.yml
                            config_CREMI_B.yml
                            config_CREMI_C.yml  
                            config_ISBI.yml
                            """)
parser.add_argument('--weights', '-w',
                    help="""The file specifying the Mod Shift setting for the weights. One of 
                            config_CREMI_A_best.yml
                            config_CREMI_B_best.yml
                            config_CREMI_C_best.yml  
                            config_ISBI_best.yml
                            """
                         )

parser.add_argument('--graph_method', '-m',
                    help="""
                            The file specifying the graph clustering method. One of 
                            config_mws_offset.yml
                            config_mws_complete.yml
                            config_kl_offset.yml
                            config_kl_complete.yml
                         """)


args = parser.parse_args()
# load data config
with open(args.data, "r") as config_file:
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)

# load weight config
with open(args.weights, "r") as config_file:
    weights_config = yaml.load(config_file, Loader=yaml.FullLoader)

# load clustering config
with open(args.graph_method, "r") as config_file:
    graph_method_config = yaml.load(config_file, Loader=yaml.FullLoader)


# load gt data for computing the scores
gt = load_data(data_config, gt = True)[:,
                                       ::graph_method_config["downsample_factor"],
                                       ::graph_method_config["downsample_factor"]]


# instantiate clustering
target_dir = os.path.join(data_config["root_path"], data_config["dataset"], "results")
if data_config["dataset"] =="CREMI":
    target_dir = os.path.join(target_dir, data_config["set"])

if graph_method_config["method"] == "MWS":
    graph_clusterer = MWS(weights_config=weights_config,
                          data_config=data_config,
                          graph_method_config=graph_method_config,
                          target_dir=target_dir)
elif graph_method_config["method"] == "KL":
    graph_clusterer = KL(weights_config=weights_config,
                         data_config=data_config,
                         graph_method_config=graph_method_config,
                         target_dir=target_dir)
else:
    raise NotImplementedError()

graph_clusterer.run(gt=gt,
                    repeats=graph_method_config["repeats"])