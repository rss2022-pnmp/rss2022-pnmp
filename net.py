"""
--
@brief:     Network class of Movement Primitives in PyTorch
"""

# Import Python libs
import math
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import deque
from nmp import util
from nmp.aggregator import *
from nmp.data_process import *
from nmp.data_assign import *
from nmp.decoder import *
from nmp.encoder import *
from nmp.loss import *
from nmp.logger import *
from nmp.mp import *


class MPNet:
    """ Movement Primitives Network """

    def __init__(self, config_dict: dict,
                 max_epoch: int,
                 init_epoch: int = 0,
                 model_api: str = None,
                 dataset_api: str = None):
        """
        Constructor

        Args:
            config_dict: configuration dictionary
            init_epoch: initialize net from this epoch if it is not 0.
            model_api: the string for load the model if init_epoch is not zero
            dataset_api: the string for loading dataset
        """

        # Create components
        self._comps = self.create_components(config_dict,
                                             dataset_api)

        # Initialize training epochs
        assert 0 <= init_epoch <= max_epoch
        self.max_epoch = max_epoch
        self.epoch = init_epoch

        # Load model if init_epoch is not 0
        if init_epoch == 0:
            assert model_api is None
        else:
            assert model_api is not None
            model_dir = self._comps["logger"]["logger"].load_model(model_api)
            self._load_weights_from_file(model_dir)

        # Initialize optimizer
        self.optimizer = \
            torch.optim.Adam(params=self._get_net_params(),
                             lr=float(self._comps["train_params"]["lr"]),
                             weight_decay=float(
                                 self._comps["train_params"]["wd"]))

        # Initialize best validation loss
        loss_deque_len = self._comps["train_params"].get("len_loss_deque", 20)
        self.loss_deque = deque(maxlen=loss_deque_len)
        self.best_validation_loss = 1e30

    @staticmethod
    def create_components(config_dict: dict,
                          dataset_api: str = None) -> dict:
        """
        Factory method to create components from config file
        Args:
            config_dict: configuration dictionary
            dataset_api: the string for loading dataset

        Returns:
            components: one dictionary containing these components:
            - logger (optional)
            - random number generator
            - dictionary of encoders,
            - aggregator,
            - agg_order,
            - decoder,
            - loss function,
            - reconstructor (optional),
            - training parameter dictionary,
            - normalizer
            - train, validate and test DataLoader,
            - data assignment manager,
        """

        # Dictionary of possible components
        ENCODER_DICT = get_encoder_dict()
        AGGREGATOR_DICT = get_aggregator_dict()
        DECODER_DICT = get_decoder_dict()
        LOSS_FUNC_DICT = get_loss_func_dict()
        LOGGER_DICT = get_logger_dict()
        DATA_ASSIGN_DICT = get_assignment_strategy_dict()

        # Initialize components as dictionary
        components = dict()

        # Dictionary of training parameters
        train_params = config_dict["train_params"]
        components["train_params"] = train_params

        # Logger
        if config_dict["logger"]["activate"] is True:

            components["logger"] = dict()

            # Initialize logger
            logger = LOGGER_DICT[config_dict["logger"]["type"]](config_dict)

            # Read log config
            watch = config_dict["logger"]["watch"]
            training_log_interval = \
                config_dict["logger"]["training_log_interval"]
            save_model_interval = config_dict["logger"]["save_model_interval"]

            # Generate log keywords dictionary
            components["logger"] = \
                {"logger": logger,
                 "watch": watch,
                 "training_log_interval": training_log_interval,
                 "save_model_interval": save_model_interval}

            # Replace local config by synchronized config
            config_dict = logger.config

        else:
            # No logger case
            components["logger"] = None

        # Random Number Generator
        if torch.cuda.is_available():
            components["rng"] = torch.Generator(device="cuda")
        else:
            components["rng"] = torch.Generator(device="cpu")
        components["rng"].manual_seed(components["train_params"]["seed"])

        # Encoders
        dict_encoders = dict()
        encoders_config = config_dict["Encoders"]
        for encoder_name, encoder_info in encoders_config.items():
            encoder_type = encoder_info["type"]
            encoder_input = encoder_info["data"]
            encoder_args = encoder_info["args"]
            encoder_args["name"] = encoder_name
            encoder = ENCODER_DICT[encoder_type](**encoder_args)
            dict_encoders[encoder_name] = {"encoder": encoder,
                                           "input": encoder_input}
        components["dict_encoders"] = dict_encoders

        # Aggregator
        aggregator_config = config_dict["Aggregator"]
        aggregator_type = aggregator_config["type"]
        aggregator_args = aggregator_config["args"]
        components["aggregator"] = \
            AGGREGATOR_DICT[aggregator_type](**aggregator_args)
        components["agg_order"] = aggregator_config["agg_order"]

        # Decoder
        decoder_config = config_dict["Decoder"]
        decoder_type = decoder_config["type"]
        decoder_args = decoder_config["args"]
        components["decoder"] = DECODER_DICT[decoder_type](**decoder_args)

        # Dataset
        dataset_config = config_dict["dataset"]
        dataset_type = dataset_config.get("type", "Pandas")
        # Load dataset from logger when necessary
        if dataset_api is not None:
            # Load dataset using logger
            assert components["logger"]["logger"] is not None
            pd_df_dict = \
                components["logger"]["logger"].load_dataset(dataset_config,
                                                            dataset_api)
        else:
            # Read dataset from local files
            seed = components["train_params"]["seed"]
            dataset_name, pd_df_dict = MPNet.read_local_dataset(dataset_config,
                                                                seed)
            # Log raw dataset when necessary
            if components["logger"] is not None and components["logger"].get(
                    "log_dataset", False):
                components["logger"]["logger"].log_dataset(
                    dataset_name=dataset_name,
                    pd_df_dict=pd_df_dict)

        if dataset_type == "Pandas":
            # PreProcessing
            transform = transforms.Compose([PreProcess.ToTensor()])

            # Generate Datasets
            train_set = MPPandasDataset(pd_df_dict["train_pd_df"],
                                        pd_df_dict["train_static_pd_df"],
                                        transform,
                                        compute_normalizer=True,
                                        **dataset_config)
            validate_set = MPPandasDataset(pd_df_dict["validate_pd_df"],
                                           pd_df_dict["validate_static_pd_df"],
                                           transform,
                                           compute_normalizer=False,
                                           **dataset_config)
            test_set = MPPandasDataset(pd_df_dict["test_pd_df"],
                                       pd_df_dict["test_static_pd_df"],
                                       transform,
                                       compute_normalizer=False,
                                       **dataset_config)
        elif dataset_type == "Mnist":
            if dataset_config.get("online_noise", False) is True:
                # PreProcessing online noise generation
                transform = \
                    transforms.Compose([PreProcess.EllipseNoiseTransform(
                        rng=components["rng"])])
            else:
                transform = None

            # Generate Datasets
            train_set = MPMnistDataset(pd_df_dict["train_images"],
                                       pd_df_dict["train_outputs"],
                                       pd_df_dict["train_trajs"],
                                       compute_normalizer=True,
                                       transform=transform,
                                       **dataset_config)
            validate_set = MPMnistDataset(pd_df_dict["validate_images"],
                                          pd_df_dict["validate_outputs"],
                                          pd_df_dict["validate_trajs"],
                                          compute_normalizer=False,
                                          transform=transform,
                                          **dataset_config)
            test_set = MPMnistDataset(pd_df_dict["test_images"],
                                      pd_df_dict["test_outputs"],
                                      pd_df_dict["test_trajs"],
                                      compute_normalizer=False,
                                      transform=transform,
                                      **dataset_config)
        else:
            raise NotImplementedError

        # Normalizer
        components["normalizer"] = train_set.get_normalizers()

        # Loss function
        loss_func = config_dict["loss_func"]["type"]
        components["loss_func"] = LOSS_FUNC_DICT[loss_func]
        components["loss_func_args"] = config_dict["loss_func"]["args"]

        components["is_mc"] = (decoder_type == "MCDecoder")
        assert components["is_mc"] != \
               (components["loss_func_args"]["num_mc_smp"] is None), \
            "Number of Monte carlo sample and decoder type not compatible!"

        # TrajectoriesReconstructor
        if config_dict["loss_func"]["type"] in get_rec_loss_func_list():
            mp_config = components["loss_func_args"]["mp_config"]
            components["reconstructor"] = \
                TrajectoriesReconstructor(mp_config)
            reconstructor_info = components["reconstructor"].get_config()
        else:
            components["reconstructor"] = None
            reconstructor_info = None

        # Dataloader
        dataloader_config = config_dict["data_loader"]
        batch_size = dataloader_config["batch_size"]
        shuffle = dataloader_config["shuffle"]
        num_workers = dataloader_config["num_workers"]

        # DataLoader
        components["train_loader"] = DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                generator=components["rng"])
        components["validate_loader"] = DataLoader(validate_set,
                                                   batch_size=len(
                                                       validate_set)//5, # todo
                                                   shuffle=shuffle,
                                                   num_workers=num_workers,
                                                   generator=components["rng"])
        components["test_loader"] = DataLoader(test_set,
                                               batch_size=len(test_set)//5,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               generator=components["rng"])

        # Data assignment manager
        data_manager_config = config_dict["data_assignment_manager"]
        manager_type = data_manager_config["type"]
        data_manager_args = data_manager_config["args"]
        data_manager_config = {"data_info": train_set.get_data_info(),
                               "encoder_info": dict_encoders,
                               "reconstructor_info": reconstructor_info,
                               "loss_info": None,
                               "normalizer": components["normalizer"],
                               "rng": components["rng"],
                               "split_info": data_manager_args}
        components["data_manager"] = \
            DATA_ASSIGN_DICT[manager_type](**data_manager_config)

        # Return
        return components

    @staticmethod
    def read_local_dataset(dataset_config,
                           random_seed):
        dataset_name = dataset_config["name"]
        dataset_type = dataset_config.get("type", "Pandas")
        if dataset_type == "Pandas":

            list_pd_df, list_pd_df_static = \
                util.read_dataset(dataset_name, shuffle=True, seed=random_seed)

            # Divide and create training, validation and testing dataset
            dataset_div = dataset_config["partition"]
            num_pd_df = len(list_pd_df)
            assert all(n > 0 for n in dataset_div.values())
            len_t = int(num_pd_df * dataset_div["train"]
                        // sum(dataset_div.values()))
            len_v = int(num_pd_df * dataset_div["validate"]
                        // sum(dataset_div.values()))

            # Dictionary of raw datasets
            pd_df_dict = \
                {"train_pd_df": list_pd_df[:len_t],
                 "train_static_pd_df": list_pd_df_static[:len_t],
                 "validate_pd_df": list_pd_df[len_t:len_t + len_v],
                 "validate_static_pd_df": list_pd_df_static[
                                          len_t:len_t + len_v],
                 "test_pd_df": list_pd_df[len_t + len_v:],
                 "test_static_pd_df": list_pd_df_static[len_t + len_v:]}
            return dataset_name, pd_df_dict

        elif dataset_type == "Mnist":
            data_dir = util.get_dataset_dir(dataset_name)

            # Get info referring to the data to be generated as torch dataset
            # images, outputs, trajs = \
            #     load_mnist_data_from_mat(data_dir,
            #                              load_original_trajectories=True)
            images, outputs, trajs = load_mnist_data_from_npz(data_dir)
            num_images = len(images)
            dataset_div = dataset_config["partition"]
            assert all(n > 0 for n in dataset_div.values())
            len_t = int(num_images * dataset_div["train"]
                        // sum(dataset_div.values()))
            len_v = int(num_images * dataset_div["validate"]
                        // sum(dataset_div.values()))
            # Dictionary of raw datasets
            data_dict = \
                {"train_images": images[:len_t],
                 "train_outputs": outputs[:len_t],
                 "train_trajs": trajs[:len_t],
                 "validate_images": images[len_t:len_t + len_v],
                 "validate_outputs": outputs[len_t:len_t + len_v],
                 "validate_trajs": trajs[len_t:len_t + len_v],
                 "test_images": images[len_t + len_v:],
                 "test_outputs": outputs[len_t + len_v:],
                 "test_trajs": trajs[len_t + len_v:]}
            return dataset_name, data_dict
        else:
            raise NotImplementedError

    def _get_net_params(self):
        """
        Get parameters to be optimized
        Returns:
             Tuple of parameters of neural networks

        """
        # Decoder
        parameters = self._comps["decoder"].parameters

        # Encoders
        for encoder_info in self._comps["dict_encoders"].values():
            parameters += encoder_info["encoder"].parameters

        # Plus decoder
        return (parameters)

    def fit(self):
        """
        External entrance of main training and validation

        Returns:
            None
        """

        # Network tuples
        networks = tuple()

        # Encoders
        for encoder_info in self._comps["dict_encoders"].values():
            networks += util.make_tuple(encoder_info["encoder"].network)

        # Decoder
        networks += util.make_tuple(self._comps["decoder"].network)

        # Watch
        if self._comps["logger"] is not None \
                and self._comps["logger"]["watch"] is True:
            log_freq = self._comps["logger"]["training_log_interval"]
            self._comps["logger"]["logger"].watch_networks(networks=networks,
                                                           log_freq=log_freq)

        # Loop over epochs
        start_epoch = self.epoch

        # Get validation interval
        validation_interval = \
            self._comps["train_params"]["validation_interval"]

        # Training loop
        for count in tqdm(range(start_epoch, self.max_epoch),
                          mininterval=10):
            if count % validation_interval == 0:
                # Train model until validate once
                validation_loss, _ = self.fit_til_validate(validation_interval)
                self.loss_deque.append(validation_loss)
                avg_validation_loss = np.asarray(self.loss_deque).mean()

                # Log average validation
                self._log_validation(avg_validation_loss)

        # Log all local stored weights
        if start_epoch != self.max_epoch \
                and self._comps["logger"] is not None:
            self._comps["logger"]["logger"].log_model(finished=True)

    def fit_til_validate(self,
                         validate_interval: int = None):
        """
        Train model until in one validate interval
        Args:
            validate_interval: N: Train N epochs and validate once

        Returns:
            validation error, self.epoch
        """

        # Make validate interval valid
        if validate_interval is None:
            validate_interval = \
                self._comps["train_params"]["validation_interval"]

        # Training
        for count in range(validate_interval):
            # Update epoch
            self.epoch += 1

            # Loss
            training_loss = 0.0

            # Mini-batch loop of training
            for num_batch_t, dict_batch_t \
                    in enumerate(self._comps["train_loader"], start=1):
                # Reset gradient
                self.optimizer.zero_grad()

                # Compute loss
                batch_loss_t = self._compute_loss(dict_batch_t)

                # Gradient back-propagation
                batch_loss_t.backward()

                # Update parameters
                self.optimizer.step()

                # Sum up batch loss
                training_loss += batch_loss_t.item()

            # todo, cpu case
            torch.cuda.empty_cache()

            # Compute average batch loss
            avg_training_loss = training_loss / num_batch_t

            # Log Training stuff
            self._log_training(avg_training_loss)

        # Validation
        with torch.no_grad():
            validation_loss = self._evaluate(self._comps["validate_loader"])

        # Return
        return validation_loss, self.epoch

    def _log_training(self,
                      avg_training_loss):
        """
        Log training data when necessary
        Args:
            avg_training_loss: training loss

        Returns:
            None
        """

        if self._comps["logger"] is not None:
            # Training loss
            if self.epoch % self._comps["logger"]["training_log_interval"] \
                    == 0 or self.epoch == self.max_epoch:
                self._comps["logger"]["logger"].log_info(self.epoch,
                                                         "Training_loss",
                                                         avg_training_loss)
            # Periodically store weights in local
            # if flag is -1, then do not log weights periodically
            if self._comps["logger"]["save_model_interval"] != -1 and \
                    (self.epoch % self._comps["logger"]["save_model_interval"]
                     == 0 or self.epoch == self.max_epoch):
                log_model_dir = self._comps["logger"]["logger"].log_model_dir
                self._write_weights_to_file(log_model_dir)

    def _log_validation(self,
                        validation_loss):
        """
        Log validation data when necessary

        Args:
            validation_loss: validation loss

        Returns:
            None
        """
        # Check if has logger
        if self._comps["logger"] is not None:
            # Validation loss
            self._comps["logger"]["logger"].log_info(self.epoch,
                                                     "Validation_loss",
                                                     validation_loss)

            # Locally store weights of model for best validation loss
            # Flag -1 indicates store best model instead of periodic one
            if self._comps["logger"]["save_model_interval"] == -1:
                # Check if best validation loss
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    log_model_dir = \
                        self._comps["logger"]["logger"].log_model_dir
                    util.remove_file_dir(log_model_dir)
                    os.makedirs(log_model_dir)
                    self._write_weights_to_file(log_model_dir)

    def _evaluate(self, data_loader):
        """
        Internal evaluate call
        Args:
            data_loader: data loader

        Returns:
            average loss
        """
        with torch.no_grad():
            loss = 0.0
            for num_batch, dict_batch in enumerate(data_loader, start=1):
                # Compute loss
                batch_loss = self._compute_loss(dict_batch)

                # Sum up batch loss
                loss += batch_loss.item()

            # Compute average batch loss
            avg_loss = loss / num_batch

            # Return
            return avg_loss

    def _compute_loss(self, dict_batch):
        """
        Compute loss
        Args:
            dict_batch: dictionary of data batch

        Returns:
            scalar loss

        """
        # Assign data
        data_manager = self._comps["data_manager"]
        data_manager.feed_data_batch(dict_batch)
        assigned_dict = data_manager.assign_data()

        dict_encoder_input = assigned_dict["dict_encoder_input"]
        decoder_input = assigned_dict["decoder_input"]
        decoder_output_ground_truth = \
            assigned_dict["decoder_output_ground_truth"]
        reconstructor_input = assigned_dict["reconstructor_input"]
        final_ground_truth = assigned_dict["final_ground_truth"]

        # Monte Carlo samples
        num_mc_smp = self._comps["loss_func_args"]["num_mc_smp"]

        # Predict
        mean_val, L_value \
            = self._predict(dict_encoder_input=dict_encoder_input,
                            decoder_input=decoder_input,
                            num_mc_smp=num_mc_smp)

        # Compute loss
        loss_kwargs = {"normalizer": self.normalizer,
                       "reconstructor": self.reconstructor,
                       "reconstructor_input": reconstructor_input,
                       "final_ground_truth": final_ground_truth,
                       "epoch": self.epoch}

        loss = self._comps["loss_func"](true_val=decoder_output_ground_truth,
                                        pred_mean=mean_val,
                                        pred_L=L_value,
                                        **loss_kwargs)

        return loss

    def _write_weights_to_file(self,
                               log_dir):
        """
        Write NN weights to files
        Args:
            log_dir: log directory to write files

        Returns:
            None

        """

        # Encoder
        for encoder_info in self._comps["dict_encoders"].values():
            encoder_info["encoder"].save_weights(log_dir, self.epoch)

        # Decoder
        self._comps["decoder"].save_weights(log_dir, self.epoch)

    def _load_weights_from_file(self,
                                log_dir):
        """
        Load NN weights from files
        Args:
            log_dir: log directory to write files

        Returns:
            None
        """
        # Encoder
        for encoder_info in self._comps["dict_encoders"].values():
            encoder_info["encoder"].load_weights(log_dir, self.epoch)
        # Decoder
        self._comps["decoder"].load_weights(log_dir, self.epoch)

    def _predict(self,
                 dict_encoder_input: dict,
                 decoder_input=None,
                 num_mc_smp: int = None):
        """
        Internal prediction call

        Args:
            dict_encoder_input: dictionary of batch data of each encoder
            decoder_input: decoder input
            num_mc_smp: num of Monte-Carlo samples when necessary

        Returns:
            a predicted tuple with mean and L
        """

        # Number of trajectories
        num_traj = None

        # Encode
        # Initialize a dictionary to store lat_obs of different encoders
        dict_lat_obs = dict()

        # Loop over all encoders
        for encoder_name, encoder_info in self._comps["dict_encoders"].items():
            # Get encoder
            encoder = encoder_info["encoder"]

            # Get data assigned to it
            context_batch = dict_encoder_input[encoder_name]
            num_traj = context_batch.shape[0]

            # Encode
            lat_obs = encoder.encode(context_batch)

            # Make it a tuple
            lat_obs = util.make_tuple(lat_obs)

            # Store latent observations
            dict_lat_obs[encoder_name] = lat_obs

        # Aggregate over different encoders
        # Reset aggregator
        self._comps["aggregator"].reset(num_traj=num_traj)

        # Firstly aggregate task parameters
        for encoder_name in self._comps["agg_order"]["default"]:
            lat_obs = dict_lat_obs[encoder_name]
            self._comps["aggregator"].aggregate(*lat_obs)

        # Then aggregate context
        for encoder_name in self._comps["agg_order"]["extra"]:
            lat_obs = dict_lat_obs[encoder_name]

            # Aggregate all context
            self._comps["aggregator"].aggregate(*lat_obs)

        # Get latent aggregation
        if self._comps["aggregator"].multiple_steps:
            lat_var = self._comps["aggregator"].get_agg_state(index=None)
        else:
            lat_var = self._comps["aggregator"].get_agg_state(index=-1)

        # Make tuple
        lat_var = util.make_tuple(lat_var)

        # Sample latent variable if necessary, MC method
        if num_mc_smp is not None:
            lat_var = self.sample_latent_variable(lat_var, num_mc_smp)

        # Decode
        mean_val, diag_cov_val, off_diag_cov_val \
            = self._comps["decoder"].decode(None, decoder_input, *lat_var)

        # De-normalize
        predict_key = self._comps["data_manager"].predict_key

        if diag_cov_val is not None:
            mean, L = \
                BatchProcess.distribution_denormalize(self.normalizer,
                                                      {predict_key: mean_val},
                                                      {predict_key:
                                                           diag_cov_val},
                                                      {predict_key:
                                                           off_diag_cov_val})
            mean = mean[predict_key]
            L = L[predict_key]
        else:
            mean = BatchProcess.batch_denormalize(self.normalizer,
                                                  {predict_key: mean_val})[
                predict_key]
            L = None

        # Return
        return mean, L

    @torch.no_grad()
    def predict(self,
                dict_obs,
                decoder_input,
                num_mc_smp: int = None):
        """
        External prediction call, as a wrapper
        Args:
            dict_obs: dictionary of observed data
            decoder_input: target times to be queried,
            num_mc_smp: Number of Monte Carlo samples, None by default

        Returns:
            predicted mean, predict covariance diagonal, and off-diagonal
        """
        # Assign data
        data_manager = self._comps["data_manager"]
        data_manager.feed_data_batch(dict_obs, inference_only=True)
        assigned_dict = data_manager.assign_data(inference_only=True)

        dict_encoder_input = assigned_dict["dict_encoder_input"]

        # Check num_mc_smp is valid or not
        num_mc_smp = self._check_monte_carlo(num_mc_smp)

        # Predict
        mean, L = self._predict(dict_encoder_input,
                                decoder_input,
                                num_mc_smp=num_mc_smp)

        # Return
        return mean, L

    def sample_latent_variable(self, lat_var,
                               num_mc_smp: int = None):
        """
        Sample latent variable for Monte-Carlo approximation
        Using re-parametrization trick

        Args:
            lat_var: tuple for mean and variance of latent variable
            num_mc_smp: num of Monte-Carlo samples when necessary

        Returns:
            sampled latent variable, shape:
            [num_traj, num_agg, num_smp, dim_lat]

            variance of latent variable, shape:
            [num_traj, num_agg, num_smp, dim_lat]
        """
        if num_mc_smp is None:
            return lat_var
        else:
            assert len(lat_var) == 2
            lat_var = list(lat_var)
            # Add one axis to MC sample-wise
            lat_var[0] = lat_var[0][..., None, :]
            lat_var[0] = lat_var[0].expand([-1, -1, num_mc_smp, -1])

            lat_var[1] = torch.sqrt(lat_var[1])[..., None, :]
            lat_var[1] = lat_var[1].expand([-1, -1, num_mc_smp, -1])

            eps = torch.randn(size=lat_var[0].shape,
                              generator=self._comps["rng"])
            # Sample lat variable from mean and std
            lat_var = (lat_var[0] + eps * lat_var[1], lat_var[1])

            # Return
            return lat_var

    @torch.no_grad()
    def use_test_dataset_to_predict(self,
                                    num_mc_smp: int = None,
                                    **kwargs):
        """
        Use test dataset to make prediction
        Args:
            num_mc_smp: number of Monte Carlo samples when necessary

        Returns:
            result_dict: a dictionary containing results for post-processing
        """
        # Get test data
        test_batch = None
        for batch in self._comps["test_loader"]:
            test_batch = batch
            break
        assert test_batch is not None

        # Compute Test Loss
        loss = self._compute_loss(test_batch)

        #  Get the key of the data to be predicted
        data_manager = self._comps["data_manager"]
        key_predict = data_manager.predict_key

        #  Compute the predicted data for the entire trajectory
        data_manager.feed_data_batch(test_batch)
        assigned_dict = data_manager.assign_data(inference_only=False,
                                                 **kwargs)

        # Get encoder input and decoder input
        dict_encoder_input = assigned_dict["dict_encoder_input"]

        # override the decoder input by full time length
        decoder_input = test_batch[key_predict]["time"] \
            if "time" in test_batch[key_predict].keys() else None
        assigned_dict["decoder_input"] = decoder_input
        del assigned_dict["original_decoder_input"]

        if "time" in test_batch[key_predict].keys():

            norm_test_batch = \
                BatchProcess.batch_normalize({key_predict:
                                                  test_batch[key_predict]},
                                                           self.normalizer)
            norm_decoder_input = norm_test_batch[key_predict]["time"]
        else:
            norm_decoder_input = None

        # Check num_mc_smp is valid or not
        num_mc_smp = self._check_monte_carlo(num_mc_smp)

        # Predict
        mean, L = self._predict(dict_encoder_input,
                                norm_decoder_input,
                                num_mc_smp=num_mc_smp)

        # Get result dictionary
        result_dict = {"test_batch": test_batch,
                       "data_manager": data_manager,
                       "loss": loss,
                       "assigned_dict": assigned_dict,
                       "mean": mean,
                       "L": L}

        return result_dict

    def _check_monte_carlo(self, num_mc_smp):
        """
        Check if the given num of Monte Carlo samples is valid
        Args:
            num_mc_smp: number of monte carlo samples

        Returns:
            num_mc_smp
        """
        if self._comps["is_mc"]:
            # Current model is Monte Carlo
            if num_mc_smp is None:
                num_mc_smp = 1
            else:
                pass
        else:
            # Current model is not Monte Carlo
            assert num_mc_smp is None, "This is not Monte Carlo case."

        # Return
        return num_mc_smp

    @property
    def normalizer(self):
        return self._comps["normalizer"]

    @property
    def reconstructor(self):
        return self._comps["reconstructor"]

    @torch.no_grad()
    def evaluate(self):
        """
        External evaluation call, evaluate model using test dataset
        Returns:
            loss on the test set
        """
        return self._evaluate(self._comps["test_loader"])

    def log_figure(self,
                   figure_obj,
                   figure_name: str = "Unnamed Figure"):
        """
        External call of logging a matplotlib figure

        Returns: None

        """
        assert self._comps["logger"]["logger"] is not None
        self._comps["logger"]["logger"].log_figure(figure_obj,
                                                   figure_name)

    def log_video(self,
                  path_to_video,
                  video_name: str = "Unnamed video"):
        """
        External call of logging a video

        Returns: None

        """
        assert self._comps["logger"]["logger"] is not None
        self._comps["logger"]["logger"].log_video(path_to_video,
                                                  video_name)

    def log_data_dict(self,
                      data_dict: dict):
        """
        Log data in dictionary
        Args:
            data_dict: dictionary to log

        Returns:
            None
        """
        assert self._comps["logger"]["logger"] is not None
        self._comps["logger"]["logger"].log_data_dict(data_dict)
