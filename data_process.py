"""
--
@brief:     Classes and method of data processing
"""

# Import Python libs
import csv
import os

import torch
import scipy.io as sio
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from addict import Dict

from nmp import util


class MPPandasDataset(Dataset):
    """
    Class of a PyTorch dataset from pandas
    """

    def __init__(self,
                 list_pd_df: list,
                 list_pd_df_static: list,
                 transform=None,
                 compute_normalizer: bool = False,
                 **kwargs):
        """
        MPPandasDataset constructor, use raw data in pandas DataFrames
        Args:
            list_pd_df: list of pandas DataFrame of time variant data
            list_pd_df_static: of pandas DataFrame of time invariant data
            transform: pre-processing functions
            compute_normalizer: if compute normalizer
            **kwargs: keyword arguments of dataset
        """

        # Get info referring to the data to be generated as torch dataset
        self.dict_data_info: dict = kwargs["data"]

        # Check if raw dataset is a list of pandas DataFrames, and have
        # all the keys referring to the data to be generated as torch dataset
        key_set = set(self.dict_data_info.keys())
        assert all(isinstance(pd_df, pd.DataFrame)
                   and isinstance(pd_df_static, pd.DataFrame)
                   and key_set.issubset(set.union(set(pd_df.columns),
                                                  set(pd_df_static.columns)))
                   for pd_df, pd_df_static in zip(list_pd_df,
                                                  list_pd_df_static))

        # Properties exist
        assert all("num_points" in info.keys()
                   and "dim_data" in info.keys()
                   and "load_func" in info.keys()
                   and "context" in info.keys()
                   and "predict" in info.keys()
                   and "time_dependent" in info.keys()
                   for info in self.dict_data_info.values())

        # Dimension of data for each key should be list or tuple
        assert all((isinstance(info["dim_data"], list)
                    or isinstance(info["dim_data"], tuple))
                   for info in self.dict_data_info.values()), \
            "Dim should be list or tuple."

        # Loading function should be callable
        assert all(callable(info["load_func"])
                   for info in self.dict_data_info.values()
                   if info["load_func"] is not None)

        # Storing
        self.transform = transform
        self.compute_normalizer = compute_normalizer

        # Get key referring to the time data
        self.key_time: str = kwargs["time"]

        # Set length of the dataset, i.e. number of pandas DataFrames
        assert len(list_pd_df) == len(list_pd_df_static)
        self.len_dataset = len(list_pd_df)

        # Initialize a dictionary to store data in cpu
        self.dict_all_data = dict()

        # Initialize a dictionary to store normalizer
        if self.compute_normalizer:
            self.dict_normalizer = dict()
        else:
            self.dict_normalizer = None

        # Finish initialization
        self._initialize(list_pd_df,
                         list_pd_df_static)

    def _initialize(self,
                    list_pd_df,
                    list_pd_df_static):
        """
        Generate dataset and normalizer
        
        Args:
            list_pd_df: list of Pandas Dataframe for time-dependent data 
            list_pd_df_static: Dataframe for time-independent data

        Returns:
            None
        """
        # Time normalizer
        if self.key_time is not None and self.compute_normalizer:
            # Get global time mean, std, min and max
            global_time_list = list()
            # Loop over all DataFrames
            for index in range(self.len_dataset):
                pd_df = list_pd_df[index]
                global_time_list.append(pd_df[self.key_time][:].values)
            global_times_array = np.concatenate(global_time_list, axis=None)

            # Store global time normalizer
            global_time_min = torch.Tensor([np.min(global_times_array)])
            global_time_max = torch.Tensor([np.max(global_times_array)])
            global_time_mean = torch.Tensor([np.mean(global_times_array)])
            global_time_std = torch.Tensor([np.std(global_times_array)])

            assert torch.min(global_time_std) > 0.0, \
                "Time should be variant rather than constant."

            self.dict_normalizer["time"] = {"min": global_time_min,
                                            "max": global_time_max,
                                            "mean": global_time_mean,
                                            "std": global_time_std}

        # Loop over all data keys
        for name, info in self.dict_data_info.items():

            # For each kind of data, get its loading function when exists
            load_func = info["load_func"]

            # Check if data is time dependent
            time_dependent = info["time_dependent"]
            num_points = info["num_points"]
            if not time_dependent:
                assert num_points == 1, \
                    "time independent variable's num_points must be 1"

            # For each kind of data, prepare numpy arrays to store data

            values = np.zeros(shape=(self.len_dataset,
                                     num_points,
                                     *info["dim_data"]))
            times = None
            if time_dependent:
                times = np.zeros(shape=(self.len_dataset,
                                        num_points,
                                        1))

            # Loop over all DataFrames
            for index in range(self.len_dataset):
                # Index DataFrame
                pd_df = list_pd_df[index]
                pd_df_static = list_pd_df_static[index]

                if time_dependent:
                    # Get data times and values referred by the name
                    pd_df_times = \
                        pd_df[self.key_time][pd_df[name].notna()].values
                    times[index] = pd_df_times.reshape(info["num_points"],
                                                       1)
                    pd_df_values = pd_df[name][pd_df[name].notna()].values
                else:
                    # Get data values referred by the name
                    pd_df_values = pd_df_static[name].values

                # Apply loading function for images
                # TODO: Apply loading function

                # Check if data is stored in an "array of string"
                if pd_df_values.dtype == object:
                    pd_df_values = np.array([util.from_string_to_array(value)
                                             for value in pd_df_values])

                # Reshape array
                values[index] = pd_df_values.reshape(info["num_points"],
                                                     *info["dim_data"])

            # Store data in dictionary
            if time_dependent:
                self.dict_all_data[name] = {"time": times, "value": values}
            else:
                self.dict_all_data[name] = {"value": values}

            if self.compute_normalizer:
                if "normalizer" in info:
                    # Specify an existing normalizer as data's normalizer
                    self.dict_normalizer[name] = \
                        self.dict_normalizer[info["normalizer"]]
                else:
                    # Compute value normalizer mean, std, min, max
                    value_mean = torch.Tensor(np.mean(values, axis=(0, 1)))
                    value_std = torch.Tensor(
                        np.std(values, axis=(0, 1), ddof=1))
                    value_min = torch.Tensor(np.min(values, axis=(0, 1)))
                    value_max = torch.Tensor(np.max(values, axis=(0, 1)))

                    assert torch.min(value_std) > 0.0, \
                        "Value should be either time variant or task variant" \
                        " rather than constant."

                    self.dict_normalizer[name] = {"mean": value_mean,
                                                  "std": value_std,
                                                  "min": value_min,
                                                  "max": value_max}

    def __len__(self):
        """
        Return the size of the dataset, i.e. number of trajectories
        Returns:
            dataset's size
        """
        return self.len_dataset

    def __getitem__(self, index):
        """
        Indexing dataset
        Args:
            index: index

        Returns:
            Dictionary of data
        """

        # Initialize a dictionary to store data
        dict_indexed_data = dict()

        # Loop over all data names
        for name, info in self.dict_data_info.items():
            if info["time_dependent"]:
                data_times = self.dict_all_data[name]["time"][index]
                data_values = self.dict_all_data[name]["value"][index]
                dict_indexed_data[name] = {"time": data_times,
                                           "value": data_values}
            else:
                data_values = self.dict_all_data[name]["value"][index]
                dict_indexed_data[name] = {"value": data_values}

        # Apply pre-processing
        if self.transform:
            dict_indexed_data = self.transform(dict_indexed_data)

        # Return dictionary of tuples
        return dict_indexed_data

    def get_normalizers(self):
        """
        Get data normalizer
        Returns:
            A dictionary storing normalizer
        """
        if self.compute_normalizer:
            return self.dict_normalizer
        else:
            raise RuntimeError("No normalizer exist!")

    def get_data_info(self):
        """
        Get data info
        Returns:
            A dict storing data info
        """
        return self.dict_data_info


def load_mnist_data_from_mat(file,
                             load_original_trajectories=False,
                             image_key='imageArray',
                             traj_key='trajArray',
                             dmp_params_key='DMPParamsArray',
                             dmp_traj_key='DMPTrajArray'):
    """
    This function is originally from the work of IMEDNET, many thanks!
    """
    # Load data struct
    data = sio.loadmat(file)

    # Parse data struct
    if 'Data' in data:
        data = data['Data']
    # Backward compatibility with old format
    elif 'slike' in data:
        data = data['slike']
        image_key = 'im'
        traj_key = 'trj'
        dmp_params_key = 'DMP_object'
        dmp_traj_key = 'DMP_trj'

    # Load images
    images = []
    for image in data[image_key][0, 0][0]:
        images.append(image.astype('float'))
    images = np.array(images)

    # Load DMPs
    DMP_data = data[dmp_params_key][0, 0][0]
    outputs = []
    for dmp in DMP_data:
        tau = dmp['tau'][0, 0][0, 0]
        w = dmp['w'][0, 0]
        goal = dmp['goal'][0, 0][0]
        y0 = dmp['y0'][0, 0][0]
        # dy0 = np.array([0,0])
        learn = np.append(tau, y0)
        # learn = np.append(learn,dy0)
        learn = np.append(learn, goal)  # Correction
        learn = np.append(learn, w)
        outputs.append(learn[..., 1:])
    outputs = np.array(outputs)

    # Load original trajectories
    original_traj = []
    if load_original_trajectories:
        trj_data = data[traj_key][0, 0][0]
        original_traj = [(trj[..., :-1]) for trj in trj_data[:]]

    return images, outputs, original_traj


def load_mnist_data_from_npz(file):
    data = np.load(file)
    images = data["images"]
    traj_x = data["traj_x"]
    traj_y = data["traj_y"]
    init_x_y = data["init_x_y"]
    dmp_w_g = data["dmp_w_g"]

    # Debug only
    # import matplotlib.pyplot as plt
    # for i in range(406, len(images)):
    #     plt.imshow(images[i, -1, 0], extent=[0, 28, 28, 0])
    #     plt.plot(traj_x[i, :], traj_y[i, :])
    #     plt.show()

    return images, np.concatenate((init_x_y, dmp_w_g), axis=-1), \
           np.stack((traj_x, traj_y), axis=-1)


class MPMnistDataset(Dataset):
    """
    Class for Mnist Pytorch Dataset
    """

    def __init__(self,
                 images,
                 outputs,
                 original_traj,
                 compute_normalizer=True,
                 transform=None,
                 **kwargs):
        self.dict_data_info: dict = kwargs["data"]
        assert {"images", "outputs", "trajs"}.issubset(kwargs["data"].keys())

        # Properties exist
        assert all("num_points" in info.keys()
                   and "dim_data" in info.keys()
                   and "context" in info.keys()
                   and "predict" in info.keys()
                   and "time_dependent" in info.keys()
                   for info in self.dict_data_info.values())

        # Dimension of data for each key should be list or tuple
        assert all((isinstance(info["dim_data"], list)
                    or isinstance(info["dim_data"], tuple))
                   for info in self.dict_data_info.values()), \
            "Dim should be list or tuple."

        # Storing
        self.transform = transform
        self.compute_normalizer = compute_normalizer

        self.key_time: str = kwargs["time"]

        # Get certain digit, e.g. 3. If None, then get all
        self.digit = kwargs.get("digit", None)

        # Initialize a dictionary to store data in cpu
        self.dict_all_data = dict()

        # Initialize a dictionary to store normalizer
        if self.compute_normalizer:
            self.dict_normalizer = dict()
        else:
            self.dict_normalizer = None

        # Finish initialization
        self._initialize(images, outputs, original_traj)

    def _initialize(self, images, outputs, trajs):

        # Time normalizer
        times = torch.linspace(start=0.0, end=3.0, steps=301)
        if self.compute_normalizer:
            self.dict_normalizer["time"] = {"min": times.min(),
                                            "max": times.max(),
                                            "mean": times.mean(-1),
                                            "std": times.std(-1)}

        if self.digit is not None:
            images_list = list()
            outputs_list = list()
            trajs_list = list()

            for d in self.digit:
                images_list.append(images[d::10])
                outputs_list.append(outputs[d::10])
                trajs_list.append(trajs[d::10])

            images = np.concatenate(images_list)
            outputs = np.concatenate(outputs_list)
            trajs = np.concatenate(trajs_list)

        self.len_dataset = len(images)

        # Debug only
        # import matplotlib.pyplot as plt
        # for i in range(1395, len(images)):
        #     plt.imshow(images[i, -1, 0], extent=[0, 41, 41, 0])
        #     plt.plot(trajs[i, :, 0], trajs[i, :, 1])
        #     plt.show()

        # Images, desired [num_traj, num_times=1, channel=1, H, W] # todo
        if images.ndim == 3:  # [num_traj, H, W]
            images = torch.Tensor(images)[:, None, None, ...].float()

        elif images.ndim == 5:  # [num_traj, num_noise_imgs = num_times, H, W]
            images = torch.Tensor(images)
        else:
            raise NotImplementedError

        self.dict_all_data["images"] = {"value": images}
        # Do not need normalizer
        if self.compute_normalizer:
            self.dict_normalizer["images"] = None

        # DMP ground-truth
        outputs = torch.Tensor(np.stack(outputs))[:, None, ...]
        self.dict_all_data["outputs"] = {"value": outputs}
        if self.compute_normalizer:
            self.dict_normalizer["outputs"] = {"mean": outputs.mean(dim=[0, 1]),
                                               "std": outputs.std(dim=[0, 1])}

        # DMP trajectories
        times = times[None, :, None].expand(self.len_dataset, -1, 1)
        trajs = torch.Tensor(np.stack(trajs))
        if self.compute_normalizer:
            self.dict_normalizer["trajs"] = {"mean": trajs.mean(dim=[0, 1]),
                                             "std": trajs.std(dim=[0, 1])}
        self.dict_all_data["trajs"] = {"time": times, "value": trajs}

        # Labels:
        # assert self.len_dataset % 10 == 0, "Label may be incorrect"
        # labels = (torch.arange(0, self.len_dataset, 1) % 10).float()[:, None,
        #          None]
        # self.dict_all_data["labels"] = {"value": labels}
        # if self.compute_normalizer:
        #     self.dict_normalizer["labels"] = {"mean": labels.mean(dim=[0, 1]),
        #                                       "std": labels.std(dim=[0, 1])}

    def __len__(self):
        """
        Return the size of the dataset, i.e. number of trajectories
        Returns:
            dataset's size
        """
        return self.len_dataset

    def __getitem__(self, index):
        """
        Indexing dataset
        Args:
            index: index

        Returns:
            Dictionary of data
        """

        # Initialize a dictionary to store data
        dict_indexed_data = dict()

        # Loop over all data names
        for name, info in self.dict_data_info.items():
            if info["time_dependent"]:
                data_times = self.dict_all_data[name]["time"][index]
                data_values = self.dict_all_data[name]["value"][index]
                dict_indexed_data[name] = {"time": data_times,
                                           "value": data_values}
            else:
                data_values = self.dict_all_data[name]["value"][index]
                dict_indexed_data[name] = {"value": data_values}

        # Apply pre-processing
        if self.transform:
            dict_indexed_data = self.transform(dict_indexed_data)

        # Return dictionary of tuples
        return dict_indexed_data

    def get_normalizers(self):
        """
        Get data normalizer
        Returns:
            A dictionary storing normalizer
        """
        if self.compute_normalizer:
            return self.dict_normalizer
        else:
            raise RuntimeError("No normalizer exist!")

    def get_data_info(self):
        """
        Get data info
        Returns:
            A dict storing data info
        """
        return self.dict_data_info


# create dictionary with parameters
cfg = Dict()

cfg.args.n_noisy = 3  # number of noisy images
cfg.noise.n_ellipses = 2
cfg.noise.radius.low = 5
cfg.noise.radius.high = 10
cfg.noise.gaussian_var = 0.25

cfg.ds.res = 40  # image resolution
cfg.ds.n_channels = 1  # number of color channels


@torch.jit.script
def create_elliptic_mask(size: int, center: torch.Tensor, radius: torch.Tensor,
                         ellip: torch.Tensor):
    """

    :param size: (scalar), e.g. x_res=y_res=32
    :param center: (n_ellipses=4, n_noisy=3, n_dim=2 (xy))
    :param radius: (n_ellipses=4, n_noisy=3)
    :param ellip:  (n_ellipses=4, n_noisy=3), ellip=1 creates a circle
    :return: (n_noisy=3, size=64, size=64])
    """
    x = torch.arange(size, dtype=torch.float32)[:, None]  # (64, 1)
    y = torch.arange(size, dtype=torch.float32)[None]  # (1, 64)

    # distance of each pixel to the ellipsis' center (4, 3, 64, 64)
    dist_from_center = torch.sqrt(
        ellip[:, :, None, None] * (x - center[:, :, 0:1, None]) ** 2
        + (y - center[:, :, 1:2, None]) ** 2 / ellip[:, :, None, None])
    # dist_from_center = torch.sqrt(ellip*(x - center[0])**2 + (y - center[1])**2/ellip)

    masks = dist_from_center <= radius[:, :, None, None]
    mask, _ = torch.max(masks, dim=0)
    return mask  # (n_noisy=3, size=64, size=64])


@torch.jit.script
def apply_mask_and_noise(mask: torch.Tensor, noise: torch.Tensor,
                         img: torch.Tensor, n_noisy: int,
                         n_channels: int):  # , translation: torch.Tensor
    imgs = img[None].repeat(n_noisy + 1, 1, 1, 1)

    if n_channels == 3:
        # apply noise and mask on all RGB color channels equally
        noise = noise.repeat(1, 3, 1, 1)
        mask = mask[:, None].repeat(1, 3, 1, 1)
    else:
        mask = mask[:, None]

    # import matplotlib.pyplot as plt
    # np_img = np.float32(mask[:, 0].transpose(0, 2))
    # np_img = np.array(imgs[2].transpose(0, 2))
    # plt.imshow(np_img)
    # plt.show()

    imgs[0:n_noisy] *= mask  # apply noise mask
    imgs[0:n_noisy] += noise  # apply additive (Gaussian) noise
    # imgs[0:n_noisy] = torch.clamp(input=imgs[0:n_noisy], min=0, max=1)
    imgs[0:n_noisy] = imgs[0:n_noisy].clamp_(min=0, max=1)
    # imgs[0:n_noisy] = torch.clamp_min(input=imgs[0:n_noisy], min=0)
    # imgs[0:n_noisy] = torch.clamp_max(input=imgs[0:n_noisy], max=1)
    return imgs


class PreProcess:
    """ A class for pre-processing when iterate dataset"""

    class ToTensor(object):
        """Convert ndarray in time-value to PyTorch Tensors."""

        def __call__(self, dict_data: dict):
            # Initialize a dictionary of time-value in PyTorch Tensors
            dict_torch = dict()

            # Convert
            # !torch.Tensor will convert to tensor in cuda if device is cuda
            # !torch.from_numpy will convert to tensor in cpu
            for name, data in dict_data.items():
                dict_temp = dict()
                for key, data_array in data.items():
                    dict_temp[key] = torch.Tensor(data_array)
                dict_torch[name] = dict_temp

            # Return
            return dict_torch

    class EllipseNoiseTransform:
        def __init__(self, rng=None):
            self.gen = rng

        def __call__(self, dict_data: dict):
            # [num_traj, num_times=1, channel=1, H, W] -> [c, h, w],  float32
            img = dict_data["images"]["value"][0]
            n_noisy = cfg.args.n_noisy

            # imgs = torch.zeros((n_noisy + 1, img.size(0), img.size(1), img.size(2)))

            radius = torch.randint(low=cfg.noise.radius.low,
                                   high=cfg.noise.radius.high,
                                   size=(cfg.noise.n_ellipses, n_noisy),
                                   generator=self.gen)
            center = torch.randint(low=1, high=cfg.ds.res - 2,
                                   size=(cfg.noise.n_ellipses, n_noisy, 2),
                                   generator=self.gen)
            ellip = torch.rand(size=(cfg.noise.n_ellipses, n_noisy),
                               generator=self.gen) + 0.5
            # translation = torch.randint(low=-cfg.noise.translation.abs, high=cfg.noise.translation.abs, size=(2, ), generator=self.gen)
            gaussian_noise = cfg.noise.gaussian_var * torch.randn(
                size=(n_noisy, 1, img.shape[1], img.shape[2]),
                generator=self.gen) if cfg.noise.gaussian_var else torch.tensor(
                0, device="cuda")
            # imgs[-1] = img
            # mask = create_elliptic_mask(size=img.shape[2], center=center,
            #                             radius=radius,
            #                             ellip=ellip)  # (n_ellipses=4, n_noisy=3, size=64, size=64])

            size = img.shape[2]
            x = torch.arange(size, dtype=torch.float32)[:, None]  # (64, 1)
            y = torch.arange(size, dtype=torch.float32)[None]  # (1, 64)

            # distance of each pixel to the ellipsis' center (4, 3, 64, 64)
            dist_from_center = torch.sqrt(
                ellip[:, :, None, None] * (x - center[:, :, 0:1, None]) ** 2
                + (y - center[:, :, 1:2, None]) ** 2 / ellip[:, :, None, None])
            # dist_from_center = torch.sqrt(ellip*(x - center[0])**2 + (y - center[1])**2/ellip)

            masks = dist_from_center <= radius[:, :, None, None]
            mask, _ = torch.max(masks, dim=0)

            transformed_img = apply_mask_and_noise(mask, gaussian_noise, img,
                                                   n_noisy,
                                                   n_channels=cfg.ds.n_channels)  # (4, 1, 64, 64)
            dict_data["images"]["value"] = transformed_img

            return dict_data


class BatchProcess:
    """ A class for processing batch data during runtime"""

    @staticmethod
    def batch_normalize(dict_batch: dict,
                        normalizer: dict):
        """
        Normalize batch data, multiple options available

        Note here the dict_batch is supposed to be a dict, for example:
        {
            "x": {"time": times, "value": values},
            "y": {"time": times, "value": values},
            "h": {"value": values}
        }

        Args:
            dict_batch: A dictionary-like raw data batch
            normalizer: A dictionary-like normalizer

        Returns:
            Normalized batch data, dictionary-like
        """

        # Initialization
        normalized_batch = dict()

        # Loop over all data keys
        for name, data_batch in dict_batch.items():

            normalized_batch[name] = dict()

            # Normalize time
            if "time" in data_batch.keys():
                time_normalizer = normalizer["time"]

                time_mean = time_normalizer["mean"]
                time_std = time_normalizer["std"]
                normalized_time = \
                    (data_batch["time"] - time_mean) / time_std

                normalized_batch[name]["time"] = normalized_time

            # Normalize value
            if "value" in data_batch.keys():
                value_normalizer = normalizer[name]
                value_mean = value_normalizer["mean"][None, None, :]
                value_std = value_normalizer["std"][None, None, :]
                normalized_value = \
                    (data_batch["value"] - value_mean) / value_std

                normalized_batch[name]["value"] = normalized_value

        # Return
        return normalized_batch

    @staticmethod
    def batch_denormalize(dict_batch: dict,
                          normalizer: dict):
        """
        Denormalize batch data, multiple options available

        Note here the dict_batch is supposed to be a dict, for example:
        {
            "x": {"time": times, "value": values},
            "y": {"time": times, "value": values},
            "h": {"value": values}
        }

        Args:
            dict_batch: A dictionary-like batch to be denormalized
            normalizer: A dictionary-like normalizer
        Returns:
            Denormalized batch data, dictionary-like
        """
        # Initialization
        denormalized_batch = dict()

        # Loop over all data keys
        for name, data_batch in dict_batch.items():

            denormalized_batch[name] = dict()

            # Denormalize time
            if "time" in data_batch.keys():
                num_add_dim = data_batch["time"].ndim - 1
                str_add_dim = str([None] * num_add_dim)[1:-1]

                time_normalizer = normalizer["time"]
                time_mean = time_normalizer["mean"][eval(str_add_dim)]
                time_std = time_normalizer["std"][eval(str_add_dim)]
                denormalized_time = data_batch["time"] * time_std + time_mean
                denormalized_batch[name]["time"] = denormalized_time

            # Denormalize value
            if "value" in data_batch.keys():
                num_add_dim = data_batch["value"].ndim - 1
                str_add_dim = str([None] * num_add_dim)[1:-1]

                value_normalizer = normalizer[name]
                value_mean = value_normalizer["mean"][eval(str_add_dim)]

                value_std = value_normalizer["std"][eval(str_add_dim)]
                denormalized_value = \
                    data_batch["value"] * value_std + value_mean
                denormalized_batch[name]["value"] = denormalized_value

        # Return
        return denormalized_batch

    @staticmethod
    def distribution_denormalize(normalizer: dict,
                                 dict_mean_batch: dict,
                                 dict_diag_batch: dict,
                                 dict_off_diag_batch: dict = None):
        """
        Denormalize predicted mean and cov Cholesky
        Args:
            normalizer: A dictionary-like normalizer
            dict_mean_batch: dict of predicted normalized mean batch
            dict_diag_batch: dict of predicted normalized diagonal batch
            dict_off_diag_batch: ... off-diagonal batch

        Returns:
            De-normalized mean and Cholesky Decomposition L

        """
        de_mean_batch = dict()
        de_L_batch = dict()
        for key in dict_mean_batch.keys():
            num_add_dim = dict_mean_batch[key].ndim - 1  # TODO: image
            str_add_dim = str([None] * num_add_dim)[1:-1]

            value_mean = normalizer[key]["mean"][eval(str_add_dim)]
            value_std = normalizer[key]["std"]

            value_std = value_std[eval(str_add_dim)]
            de_norm_matrix = util.build_lower_matrix(value_std, None)

            # Denormalize mean
            de_mean_batch[key] = torch.einsum('...ij,...j->...i',
                                              de_norm_matrix,
                                              dict_mean_batch[key]) + value_mean

            # Denormalize cov Cholesky
            if dict_off_diag_batch is None:
                L = util.build_lower_matrix(dict_diag_batch[key],
                                            None)
            else:
                L = util.build_lower_matrix(dict_diag_batch[key],
                                            dict_off_diag_batch[key])

            de_L_batch[key] = torch.einsum('...ij,...jk->...ik',
                                           de_norm_matrix,
                                           L)

        return de_mean_batch, de_L_batch


def compound_data(dataset: str,
                  compound_dynamic_names: list = None,
                  compound_static_names: list = None,
                  compound_name: str = None,
                  to_dataset: str = None):
    list_pd_df, list_pd_df_static = util.read_dataset(dataset)
    dataset_path = util.get_dataset_dir(to_dataset)
    util.remove_file_dir(dataset_path)
    os.makedirs(dataset_path)
    for i, (pd_df, pd_df_static) in enumerate(zip(list_pd_df,
                                                  list_pd_df_static)):
        if compound_dynamic_names is not None:
            dynamic_data_list = list()
            for dynamic_name in compound_dynamic_names:
                data = pd_df[dynamic_name].values
                if data.dtype == object:
                    data = util.from_string_to_array(data)
                dynamic_data_list.append(data)
            compounded_data = list(np.stack(dynamic_data_list, axis=-1))
            if compound_name is None:
                compound_name = '_'.join(compound_dynamic_names)
            pd_df[compound_name] = compounded_data
            pd_df.to_csv(path_or_buf=dataset_path + "/" + str(i) + ".csv",
                         index=False,
                         quoting=csv.QUOTE_ALL)

        if compound_static_names is not None:
            pass
