import argparse
import sys
sys.path.append("..")
import yaml
from utils.dataloader import load_data
from utils.clusterers import get_clusterer
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
parser.add_argument('--clustering', '-c',
                    help="""The configuration file of the clustering. Must be one of 
                            config_single_linkage.yml
                            config_complete_linkage.yml
                            config_hdbscan.yml
                            config_flat_mean_shift.yml
                            config_flat_blurring_mean_shift.yml
                            config_gaussian_mean_shift.yml
                            config_gaussian_blurring_mean_shift.yml
                            config_mod_shift.yml
                            config_blurring_mod_shift.yml
                            """
                         )

args = parser.parse_args()
# load data config
with open(args.data, "r") as config_file:
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)

# load data and gt
data = load_data(data_config)
gt = load_data(data_config, gt = True)

# load clustering config
with open(args.clustering, "r") as config_file:
    clustering_config = yaml.load(config_file, Loader=yaml.FullLoader)



# instantiate clustering
target_dir = os.path.join(data_config["root_path"], data_config["dataset"], "results")
if data_config["dataset"] =="CREMI":
    target_dir = os.path.join(target_dir, data_config["set"])

clusterer = get_clusterer(clustering_config["method"],
                          clustering_config,
                          target_dir=target_dir,
                          dataset= data_config["dataset"])

# perform clustering
best_results = clusterer.run(data, gt)

print("Best results for ")
for parameter in best_results["parameters"]:
    print("{}: {}".format(parameter, best_results["parameters"][parameter]))
print(best_results["scores"])