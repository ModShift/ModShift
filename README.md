# ModShift reproducability

## How to use this repository

1. Install Anaconda following [these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Clone the repository `git clone https://github.com/ModShift/ModShift.git`
3. Create and activate the environment
   ```
   cd ModShift
   conda env create -f environment.yml
   conda activate ModShift
   ```
4. Download the data automatically `python dataloader.py` or manually
   * For  CREMI  download  the  file  from [here](https://drive.google.com/file/d/1eOPfoKXmDPnxt_hibRjCMN4K8c4jhYWO/view) and place it in `ModShift/data/CREMI/data/CREMI.h5`.
   * For ISBI download the file from [here](https://drive.google.com/file/d/1E_OqBdOqEIfrK19H4gOxN2qkGYNbAknR/view) and place it in `ModShift/data/ISBI/data/ISBI.h5`.
   * For ISBI's PCA download the file from [here](https://drive.google.com/open?id=1r5n8ReXsJZXk0xrNsPJZ01SJUCzFRV9E) and place it in `ModShift/data/ISBI/data/ISBI_embeddings_PCA_8.npy`.
5. Reproduce  the  pixel  embedding-based  clustering  results by running
   ```
   python main.py --data config_DATASET.yml --clustering config_CLUSTERING_METHOD.yml
   ```
   For instance, to reproduce fixed Mod Shift’s results on the subvolume of CREMI A run
   ```
   python main.py --data config_CREMI_A.yml --clustering config_mod_shift.yml
   ```
   Convergence points, labels and scores will be saved in `data/DATASET/`. The best parameters and scores will be printed.
   In case of problems with PyKeOps, confer [here](https://www.kernel-operations.io/keops/python/installation.html). Note that the pure PyTorch implementation is very slow on these datasets, so that downsampling them in the `config_DATASET.yml` files is advisable if PyKeOps is not available.
6. To reproduce the experiment of section D.3 do 
   ```
   cd uniform_comparison
   python mean_mod_uniform.py
   ```
   The key statistics will be printed, the number of clusters for Mean Shift and Mod Shift saved and the histogram of these numbers of clusters plotted.
7. To reproduces an experiment on the toy dataset, do
   ```
   cd 3clusters
   python trajectories_toy.py
   ```
   The trajectories will be saved in `ModShift/3clusters/data/3clusters/results` and plotted.
  
 ## License 
 This repository is licensed under the MIT License other than the directory cremi, which is strongly based on and licensed as [this repository](https://github.com/cremi/cremi_python).
