import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mrinet.unet_wavelet import UNetWavelet
from data.metrics import MetricEval


class DeGibbs(object):
    """A U-Net model for Gibbs artifact correction.

    This initializes a wavelet U-Net model intended for artifact correciton in
    medical images. The script allows attachment of PyTorch data loader
    objects and associated transforms, in principle allowing the correction of
    any artifacts simulated by the transforms.

    By default, the model output parameters are saved in
    'exp_dir/best_model.pt'

    Args:
        nlayers (int): Number of layers for U-Net.
        in_ch (int): Number of input channels (usually 1 for magnitude, 2 for
            complex).
        out_ch (int): Number of output channels.
        comp2mag (boolean): A flag for whether to output magnitude outputs when
            given complex inputs. If True, then performs the magnitude operation
            as a final step of the U-Net.
        leaky (boolean): If true, use leaky ReLUs instead of normal ones.
        device (torch.device): Use torch.device('cuda') for GPU or
            torch.device('cpu') for CPU.

    Examples:
        Initialization:

        >>> ob = DeGibbs(**params)

        Loading parameters from file:

        >>> ob = ob.load_model(model_file)

        Attaching a dataloader:

        >>> ob = ob.attach_data(exp_type, train_loader, val_loader)

        Training:

        >>> ob = ob.fit(num_epochs)

        Run current model on example:

        >>> out = ob(in)
    """

    def __init__(
        self,
        nlayers,
        in_ch,
        out_ch,
        nchans,
        comp2mag,
        leaky,
        device,
        dtype=torch.double,
    ):
        self.optimizer = None
        self.train_data_dir = None
        self.model = UNetWavelet(
            ndims=2,
            nlayers=nlayers,
            in_ch=in_ch,
            out_ch=out_ch,
            top_filtnum=nchans,
            resid=True,
            wave_concat=True,
            comp2mag=comp2mag,
            leaky=leaky,
        )

        self.model = self.model.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return self.model(x)

    def load_model(self, model_file, device=None, dtype=None):
        """Loads model parameters from .pt file.

        Args:
            model_file (str): A directory pointing to the .pt file with
                parameters.
            device (torch.device, default=None): Device to send model to after
                loading. If None, uses device supplied by init function.

        Returns:
            self
        """
        if not device == None:
            self.device = device
            self.model = self.model.to(device)
        if not dtype == None:
            self.dtype = dtype

        self.model = self.model.to(dtype=dtype)

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("network params: {}".format(pytorch_total_params))

        if os.path.isfile(model_file):
            params = self.model.state_dict()

            print("loading model from file: {}".format(model_file))
            checkpt = torch.load(model_file, map_location=self.device)
            state_dict = checkpt["state_dict"]
            params.update(state_dict)
            self.model.load_state_dict(params)
        else:
            print("model file {} not found".format(model_file))

        self.model = self.model.eval()

        return self

    def init_optimizer(self, learning_rate, loss_fn, model_file=None):
        """Loads model parameters from .pt file.

        Currently, only the PyTorch Adam optimizer is implemented.

        Args:
            learning_rate (double): The Adam learning rate.
            loss_fn (torch.loss_fn): The PyTorch loss function
                (e.g., torch.nn.MSELoss).
            model_file (str): If not None, then loads the optimizer state from
                the model file.

        Returns:
            self
        """
        self.epoch = 0
        self.train_loss_min = np.inf
        self.val_loss_min = np.inf
        self.loss_fn = loss_fn

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if os.path.isfile(model_file) and model_file is not None:
            print("loading optimizer from file: {}".format(model_file))
            checkpt = torch.load(model_file)

            optimizer.load_state_dict(checkpt["optimizer"])

            self.epoch = checkpt["epoch"] + 1
            train_loss_min = self.train_loss_min
            try:
                self.train_loss_min = checkpt["train_loss_min"]
            except:
                self.train_loss_min = train_loss_min
            val_loss_min = self.val_loss_min
            try:
                self.val_loss_min = checkpt["val_loss_min"]
            except:
                self.val_loss_min = val_loss_min
        elif not os.path.isfile(model_file):
            print("model file {} not found".format(model_file))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUS!")
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = optimizer
        self.learning_rate = learning_rate

        return self

    def attach_data(self, exp_type, train_loader, val_loader, train_data_dir=None):
        """Attaches a dataloader for training.

        Args:
            exp_type (str): Experiment type (used for visualizations).
            train_loader (torch.DataLoader): The loader for the training split.
            val_loader (torch.DataLoader): The loader for the validation split.
            train_data_dir (str, default=None): Stored as attribute for print
                statements.

        Returns:
            self
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_type = exp_type
        self.train_data_dir = train_data_dir

        return self

    def _save_checkpoint(self, filename):
        """Save current model state.

        Args:
            filename (str): File name for .pt file to save model.
        """
        try:
            state = {
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
        except AttributeError:
            state = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

        for attr, value in self.__dict__.items():
            if (
                (not "model" in attr)
                and (not "optimizer" in attr)
                and (not "loss_fn" in attr)
            ):
                state[attr] = value

        torch.save(state, filename)

    def run_val_loop(self, visuals, rnd_seed, all_metrics=False):
        """Run a loop over the validation data split.

        Args:
            visuals (dict): A dictionary object storing all arrays for
                visualization.
            rnd_seed (int): Seed for numpy, PyTorch, and CUDA (for
                reproducibility).

        Returns:
            val_loss (double): Validation loss.
            visuals (dict): Dictionary visuals object with validation images.
        """
        print("epoch {}: validation loop".format(self.epoch))
        # set random seed
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.cuda.manual_seed(rnd_seed)
        device = self.device

        self.model = self.model.eval()

        disp_frac_vec = np.array(range(11))
        disp_val = disp_frac_vec[0]
        disp_counter = 0

        if all_metrics:
            metrics = {"mse": [], "ssim": [], "nmse": [], "psnr": []}

        with torch.no_grad():
            val_loss = 0
            counter = 0
            for i, val_batch in enumerate(self.val_loader):
                val_target, val_dat = (
                    val_batch["target"].to(device),
                    val_batch["dat"].to(device),
                )

                val_est = self.model(val_dat)
                val_loss = (
                    val_loss * counter + self.loss_fn(val_est, val_target).item()
                ) / (counter + 1)
                counter = counter + 1

                if all_metrics:
                    for ind in range(val_target.shape[0]):
                        nptarget = val_target[ind].cpu().numpy()
                        if nptarget.shape[0] > 1:
                            nptarget = np.sqrt(nptarget[0] ** 2 + nptarget[1] ** 2)
                        else:
                            nptarget = nptarget[0]
                        npest = val_est[ind].cpu().numpy()
                        if npest.shape[0] > 1:
                            npest = np.sqrt(npest[0] ** 2 + npest[1] ** 2)
                        else:
                            npest = npest[0]
                        metrics["mse"].append(MetricEval.mse(nptarget, npest))
                        metrics["ssim"].append(MetricEval.ssim(nptarget, npest))
                        metrics["nmse"].append(MetricEval.nmse(nptarget, npest))
                        metrics["psnr"].append(MetricEval.psnr(nptarget, npest))

                if (i / len(self.val_loader)) >= (disp_val / 10):
                    print(
                        "validation loop progress: {:.0f}%".format(
                            100 * (i + 1) / len(self.val_loader)
                        )
                    )
                    disp_counter += 1
                    disp_val = disp_frac_vec[disp_counter]

            print("validation loop finished")

            dispind = 0

            if val_target.shape[1] > 1:  # assume dim 0, 1 are real, imag
                tmp = np.squeeze(
                    val_target[dispind, ...].cpu().detach().float().numpy()
                )
                visuals["val_target"] = np.expand_dims(
                    np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
                )
            else:
                visuals["val_target"] = np.absolute(
                    val_target[dispind, ...].cpu().detach().numpy()
                )
            if val_dat.shape[1] > 1:  # assume dim 0, 1 are real, imag
                tmp = np.squeeze(val_dat[dispind, ...].cpu().detach().float().numpy())
                visuals["val_dat"] = np.expand_dims(
                    np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
                )
            else:
                visuals["val_dat"] = np.absolute(
                    val_dat[dispind, ...].cpu().detach().numpy()
                )
            if val_est.shape[1] > 1:  # assume dim 0, 1 are real, imag
                tmp = np.squeeze(val_est[dispind, ...].cpu().detach().float().numpy())
                visuals["val_est"] = np.expand_dims(
                    np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
                )
            else:
                visuals["val_est"] = np.absolute(
                    val_est[dispind, ...].cpu().detach().float().numpy()
                )

        if all_metrics:
            mse = np.array(metrics["mse"])
            mse = mse[~np.isnan(mse)]
            mse = mse[~np.isinf(mse)]
            nmse = np.array(metrics["nmse"])
            nmse = nmse[~np.isnan(nmse)]
            nmse = nmse[~np.isinf(nmse)]
            ssim = np.array(metrics["ssim"])
            ssim = ssim[~np.isnan(ssim)]
            ssim = ssim[~np.isinf(ssim)]
            psnr = np.array(metrics["psnr"])
            psnr = psnr[~np.isnan(psnr)]
            psnr = psnr[~np.isinf(psnr)]

            metrics["mse"] = mse
            metrics["nmse"] = nmse
            metrics["ssim"] = ssim
            metrics["psnr"] = psnr

            return metrics, visuals
        else:
            return val_loss, visuals

    def run_train_loop(self, global_iter, visuals, rnd_seed):
        """Run a loop over the training data split.

        The model is updated via the self.model attribute.

        Args:
            global_iter (int): Global iteration count (for print statements).
            visuals (dict): A dictionary object storing all arrays for
                visualization.
            rnd_seed (int): Seed for numpy, PyTorch, and CUDA (for
                reproducibility).

        Returns:
            global_iter (int): Updated global iteration.
            visuals (dict): Dictionary visuals object with validation images.
            disp_loss (double): A display loss for the training data taken as
                an average from the last few images.
        """
        print("epoch {}: training loop".format(self.epoch))
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.cuda.manual_seed(rnd_seed)

        device = self.device

        self.model = self.model.train()
        losses = []

        disp_frac_vec = np.array(range(11))
        disp_val = disp_frac_vec[0]
        disp_counter = 0

        for i, batch in enumerate(self.train_loader):
            target, dat = batch["target"].to(device), batch["dat"].to(device)

            est = self.model(dat)
            loss = self.loss_fn(est, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            np_train_loss = loss.item()
            losses.append(np_train_loss)
            losses = losses[-50:]
            disp_loss = np.mean(losses)

            global_iter = global_iter + 1

            if (i / len(self.train_loader)) >= (disp_val / 10):
                print(
                    "training loop progress: {:.0f}%".format(
                        100 * (i + 1) / len(self.train_loader)
                    )
                )
                disp_counter += 1
                disp_val = disp_frac_vec[disp_counter]

        print("training loop finished")

        dispind = 0

        if target.shape[1] > 1:  # assume dim 0, 1 are real, imag
            tmp = np.squeeze(target[dispind, ...].cpu().detach().float().numpy())
            visuals["train_target"] = np.expand_dims(
                np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
            )
        else:
            visuals["train_target"] = np.absolute(
                target[dispind, ...].cpu().detach().numpy()
            )
        if dat.shape[1] > 1:  # assume dim 0, 1 are real, imag
            tmp = np.squeeze(dat[dispind, ...].cpu().detach().float().numpy())
            visuals["train_dat"] = np.expand_dims(
                np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
            )
        else:
            visuals["train_dat"] = np.absolute(dat[dispind, ...].cpu().detach().numpy())
        if est.shape[1] > 1:  # assume dim 0, 1 are real, imag
            tmp = np.squeeze(est[dispind, ...].cpu().detach().float().numpy())
            visuals["train_est"] = np.expand_dims(
                np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
            )
        else:
            visuals["train_est"] = np.absolute(
                est[dispind, ...].cpu().detach().float().numpy()
            )

        return global_iter, visuals, disp_loss

    def fit(self, num_epochs, exp_dir=None, run_eval=True, seed_offset=476):
        """Fit attached data loaders (i.e., train the model).

        The model is updated via the self.model attribute.

        Args:
            num_epochs (int): Number of epochs to train.
            exp_dir (str, default=None): String pointing to directory for
                logging tensorboardX outputs.
            run_eval (boolean, default=True): Whether to run validation loop.
            seed_offset (int, default=476): Offset for random number seed (for
                reproducibility).

        Returns:
            self
        """
        print("starting training")

        if not exp_dir == None:
            print("saving logs to {}".format(exp_dir))
            writer = SummaryWriter(log_dir=exp_dir)
        visuals = {}

        print("initializing loss tracking")
        epochs = []
        train_losses = []
        val_losses = []

        global_iter = 0

        for epoch in range(self.epoch, num_epochs):
            rnd_seed = np.random.get_state()[1][0] + self.epoch + seed_offset
            self.epoch = epoch
            epoch_start = time.time()

            global_iter, visuals, train_loss = self.run_train_loop(
                global_iter, visuals, rnd_seed
            )

            if run_eval:
                val_loss, visuals = self.run_val_loop(visuals, seed_offset)

                if not exp_dir == None:
                    writer.add_scalar(
                        "losses/eval_loss", scalar_value=val_loss, global_step=epoch
                    )

            if not exp_dir == None:
                writer.add_scalar(
                    "losses/train_loss", scalar_value=train_loss, global_step=epoch
                )

                for label, image in visuals.items():
                    if "val" in label:
                        image = image / np.max(image)
                        writer.add_image(
                            "validation/" + label, image, global_step=epoch
                        )
                    elif "train" in label:
                        image = image / np.max(image)
                        writer.add_image("training/" + label, image, global_step=epoch)

            if run_eval:
                val_losses.append(val_loss)
                if val_loss < self.val_loss_min:
                    self.val_loss_min = val_loss
                    checkname = os.path.join(exp_dir, "best_model.pt")
                else:
                    checkname = os.path.join(exp_dir, "model_epoch_{}.pt".format(epoch))
            else:
                if train_loss < self.train_loss_min:
                    checkname = os.path.join(exp_dir, "best_model.pt")
                else:
                    checkname = os.path.join(exp_dir, "model_epoch_{}.pt".format(epoch))

            if train_loss < self.train_loss_min:
                self.train_loss_min = train_loss

            train_losses.append(train_loss)
            epochs.append(epoch)

            self._save_checkpoint(checkname)

            epoch_end = time.time()

            print("epoch finished, time: {}".format(epoch_end - epoch_start))
            print("validation loss: {}, training loss: {}".format(val_loss, train_loss))

        return self

    def __repr__(self):
        self.model = self.model.train()
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        out = "\n" + self.__class__.__name__ + "\n"
        out += "model: {}".format(self.model.__class__.__name__)
        out += "    number of trainable model parameters: {}".format(num_params)
        out += "optimizer: {}".format(self.optimizer.__class__.__name__)
        out += "train_data_dir: {}".format(self.train_data_dir)

        return out
