import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")


# implements a class for a mod shift clustering and the corresponding model which
# computes the objective function of mod shift

# implements a mod shift instance
class ModShift(object):
    def __init__(self, data, config):
        # expects data in the form ? * channels
        np.random.seed(42)

        self.device = torch.device(config["torch"]["device"])
        self.dtype = torch.float
        self.config = config
        self.blurring = self.config["optimization"]["blurring"]

        # reshape data from B(atch)* ? * E(mbd dim) to batch_size * channels (embd_dim) * ?
        data = np.moveaxis(data, -1, 1)

        self.points0 = torch.from_numpy(data).to(device = self.device, dtype = self.dtype)
        self.points0.requires_grad = False
        self.mv_points = torch.from_numpy(data).to(device = self.device, dtype = self.dtype)
        self.mv_points.requires_grad = True
        self.batch_size = config["optimization"]["batch_size"]
        self.batch_size_clipped = np.minimum(self.batch_size, 1024).astype(int) # make sure large batches work out!!!
        self.learning_rate = config["optimization"]["optimizer"]["learning_rate"]
        self.scheduler_ratio = config["optimization"]["scheduler"]["ratio"]
        self.scheduler_gamma = config["optimization"]["scheduler"]["gamma"]

        self.momentum = config["optimization"]["optimizer"]["momentum"]

        # initialise optimiser, only mv_points are updated, even for blurring
        if config["optimization"]["optimizer"]["type"] == "Adam":
            self.optimizer = torch.optim.Adam([self.mv_points], lr = self.learning_rate)
        elif config["optimization"]["optimizer"]["type"]  == "SGD":
            self.optimizer = torch.optim.SGD(params=[self.mv_points], lr=self.learning_rate, momentum = self.momentum, nesterov = False)
        else: print("Please specify available optimiser.")

        # initialise dataset and dataloader
        self.dataset = TensorDataset(self.mv_points, self.points0)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle = True)

        if not config["keops"]:
            from .ModShiftModel import ModShiftModel
            self.model = ModShiftModel(config["rho"], config["weight_func"], config)
        elif config["keops"]:
            from .ModShiftModel_keops import ModShiftModel
            self.model = ModShiftModel(config["rho"], config["weight_func"], config)

        # initialise lists to save intermediate results
        self.loss_list = []
        self.trajectories = [np.copy(self.mv_points.cpu().data.numpy())]


    def run(self, num_epochs):
        # set up scheduler
        if isinstance(self.scheduler_ratio, float):
            # this is the one-time lr decrease
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = np.floor(num_epochs * self.scheduler_ratio), gamma = self.scheduler_gamma)
        elif isinstance(self.scheduler_ratio, list):
            # several-time lr decrease
            milestones = [np.floor(num_epochs * ratio) for ratio in self.scheduler_ratio]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                             milestones=milestones,
                                                             gamma=self.scheduler_gamma)
        else: print("No valid scheduler ratio is provided.")


        # do the mod shifting
        for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
            loss_epoch = 0
            for mv_points_batch, points0_batch in self.dataloader:

                # compute gradients without keops with tricks keeping the GPU-memory footprint manageable
                if self.config["keops"] == False:
                    from .ModShiftModel import compute_model
                    loss_batch = compute_model(mv_points_batch,
                                                           points0_batch,
                                                           self.model,
                                                           self.config)


                    if self.config["optimization"]["compute_gradients_early"] == False:
                        loss_epoch += loss_batch.detach()
                        loss_batch.backward()
                    else:
                        #gradients are already computed in compute_model
                        loss_epoch += loss_batch

                # compute gradients using keops
                elif self.config["keops"] == True:
                    if self.blurring:
                        loss_batch = self.model(mv_points_batch)
                    else:
                        loss_batch = self.model(mv_points_batch, points0_batch)

                    loss_epoch += loss_batch.detach()
                    loss_batch.sum().backward()

                else: print("keops option can only be True or False")

                # perform a gradient update
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.loss_list.append(loss_epoch.cpu().numpy())

            if self.config["save_all"]:
                self.trajectories.append(np.copy(self.mv_points.cpu().data.numpy()))

            scheduler.step()

            if epoch%np.ceil(num_epochs/100) == 0: tqdm.write("Epoch {} is done ".format(epoch))

        if not self.config["save_all"]:
            self.trajectories.append(np.copy(self.mv_points.cpu().data.numpy()))
        self.trajectories = np.array(self.trajectories)
