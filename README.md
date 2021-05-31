# ModShift reproducability

## How to use this repository

1. Install Anaconda following [these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Clone the repository `git clone https://github.com/ModShift/ModShift.git`
3. Create and activate the environment
   ```
   cd test_reprod
   conda env create -f environment.yml
   conda activate ModShift
   ```
4. Download the pixel embedding data automatically `python dataloader.py` or manually
   * For  CREMI  download  the  file  from [here](https://drive.google.com/file/d/1eOPfoKXmDPnxt_hibRjCMN4K8c4jhYWO/view) and place it in `ModShift/data/CREMI/data/CREMI.h5`.
   * For ISBI download the file from [here](https://drive.google.com/file/d/1E_OqBdOqEIfrK19H4gOxN2qkGYNbAknR/view) and place it in `ModShift/data/ISBI/data/ISBI.h5`.
   * For ISBI's PCA download the file from [here](https://drive.google.com/open?id=1r5n8ReXsJZXk0xrNsPJZ01SJUCzFRV9E) and place it in `ModShift/data/ISBI/data/ISBI_embeddings_PCA_8.npy`.
   
We obtained the CREMI raw data from [here](https://cremi.org/), where it is licensed under the CC-BY license.
We obtained the ISBI raw data  from [here](http://brainiac2.mit.edu/isbi_challenge/home). It is licensed as follows:

   'You are free to use this data set for the purpose of generating or testing non-commercial image segmentation software. If any scientific publications derive from the usage of this data set, you must cite [TrakEM2](http://t2.ini.uzh.ch/trakem2.html) and the following publication:

Cardona A, Saalfeld S, Preibisch S, Schmid B, Cheng A, Pulokas J, Tomancak P, Hartenstein V. 2010. An Integrated Micro- and Macroarchitectural Analysis of the Drosophila Brain by Computer-Assisted Serial Section Electron Microscopy. PLoS Biol 8(10): e1000502. doi:10.1371/journal.pbio.1000502.'
     
5. Reproduce  the  pixel  embedding-based  clustering  results by running
   ```
   python main.py --data config_DATASET.yml --clustering config_CLUSTERING_METHOD.yml
   ```
   For instance, to reproduce non-blurring Mod Shift’s results on the subvolume of CREMI A run
   ```
   python main.py --data config_CREMI_A.yml --clustering config_mod_shift.yml
   ```
   Convergence points, labels and scores will be saved in `data/DATASET/`. The best parameters and scores will be printed.
   In case of problems with PyKeOps, confer [here](https://www.kernel-operations.io/keops/python/installation.html). Note that the pure PyTorch implementation is very slow on these datasets, so that downsampling them in the `config_DATASET.yml` files is advisable if PyKeOps is not available.
6. Recompute the run times for representative experiments with 
   ```
   python main_runtimes.py
   ```
7. Recompute the graph clustering baselines with
   ```
   python main_graph.py -d config_DATASET.yml -w config_DATASET_best.yml -m config_GRAPHMETHOD_complete.yml
   ```
   Replace `DATASET` as above and `GRAPHMETHOD` with either `kl` for Kernighan-Lin approximated Multicut or with `mws` 
   for the Mutex Watershed.
8. To reproduce the experiment of section D.3 do 
   ```
   cd uniform_comparison
   python mean_mod_uniform.py
   ```
   The key statistics will be printed, the number of clusters for Mean Shift and Mod Shift saved and the histogram of these numbers of clusters plotted.
9. To reproduces an experiment on the toy dataset, do
   ```
   cd 3clusters
   python trajectories_toy.py
   ```
   The trajectories will be saved in `ModShift/3clusters/data/3clusters/results` and plotted.
  
 ## License 
 This repository is licensed under the MIT License other than the directory cremi, which is strongly based on [this repository](https://github.com/cremi/cremi_python).
