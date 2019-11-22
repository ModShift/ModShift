import sys
sys.path.append("..")
from mod_shift.ModShift import ModShift
import yaml
from utils.dataloader import load_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib.colors as colors
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# load config
with open("config_trajectories_toy.yml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)


# load data
data = np.load(os.path.join(config["data"]["root_path"],
                            "3clusters",
                            "data",
                            "3clusters.npy"))[None, ::config["data"]["downsample_factor"], :]  # 1 N 2


# perform and save clustering
ModShifter = ModShift(data, config["ModShift"])

ModShifter.run(config["ModShift"]["optimization"]["iterations"])

trajectories = ModShifter.trajectories

target_dir = os.path.join(config["data"]["root_path"], config["data"]["dataset"], "results")

if not os.path.exists(target_dir):
    try:
        os.makedirs(target_dir)
    except OSError:
        print("Creation of the directory %s failed" % target_dir)
np.save(os.path.join(target_dir, "mod_shift_trajectories.npy"), trajectories)

# plot trajectories
fig = plt.figure(figsize=(8.0, 6.0))
axs = fig.add_subplot(1,1,1)

# for logarithmic colorbar
norm = colors.LogNorm(1, trajectories.shape[0]) #- max_time % step_size_plot+1)
plt.tick_params(labelsize=20)

# iterate over all initial points and plot corresponding converged points for
for i in range(trajectories.shape[-1]):
    points = np.array([trajectories[:, 0, 0, i], trajectories[:,0, 1, i]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segments = np.concatenate([np.tile(trajectories[0, 0, :, i].reshape(2, 1), (1, 2)).T[None, ...], segments], axis = 0)

    lc = LineCollection(segments, cmap='jet', norm=norm)

    lc.set_array(np.arange((trajectories.shape[0]-1)))

    line = axs.add_collection(lc)


# scatter initial and final points
initial_1 = axs.scatter(trajectories[0, 0, 0, :10], trajectories[0, 0, 1, :10],
            color= "k", zorder = 5, marker = "^", label= "data points")
initial_2 = axs.scatter(trajectories[0, 0, 0, 10:20], trajectories[0, 0, 1, 10:20],
            color= "k", zorder = 5, marker = "P", label= "data points")
initial_3 = axs.scatter(trajectories[0, 0, 0, 20:30], trajectories[0, 0, 1, 20:30],
            color= "k", zorder = 5, marker = "v", label= "data points")
conv_1 = axs.scatter(trajectories[-1, 0, 0, :10], trajectories[-1, 0, 1, :10],
            color= "red", zorder = 5, marker = "^", label="convergence \npoints")
conv_2 = axs.scatter(trajectories[-1, 0, 0, 10:20], trajectories[-1, 0, 1, 10:20],
            color= "red", zorder = 5, marker = "P", label="convergence \npoints")
conv_3 = axs.scatter(trajectories[-1, 0, 0, 20:30], trajectories[-1, 0, 1, 20:30],
            color= "red", zorder = 5, marker = "v", label="convergence \npoints")


l1 = plt.legend([(conv_1, conv_2, conv_3), (initial_1, initial_2, initial_3)], ["convergence \npoints", "data points"],
               numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)},
               fontsize = 20.)  # , loc=[0.2, 0.3])


plt.ylim(-0.4, 1.4)
plt.xlim(-0.6, 2.6)

plt.gca().set_aspect(aspect = "equal")

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.4)

cbar = fig.colorbar(line, cax=cax, orientation = "horizontal")
cbar.set_label("time", size = 20.)
cbar.ax.tick_params(labelsize = 20.)

# save with the correct title
plt.savefig(os.path.join(target_dir, "toy_mod_shift_many.pdf"), bbox_inches="tight", pad_inches=0)

plt.show()
