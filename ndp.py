"""
@brief:     Network class of Neural Dynamic Policies (NDP) in PyTorch
@details:   Due to the lack of maintenance and code documentation, the original
implementation of NDP cannot be used unfortunately, so we re-implement NDP,
based on our best knowledge to their work. We utilize a lot of helper
functions shared from NMP.
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
from nmp.net import *
from nmp.mp import *


class NDP:
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
            - network
            - loss function,
            - reconstructor
            - training parameter dictionary,
            - normalizer
            - train, validate and test DataLoader,
            - data assignment manager,
        """
        ENCODER_DICT = get_encoder_dict()
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

        # NDP Network
        dict_networks = dict()
        network_config = config_dict["Networks"]
        for network_name, network_info in network_config.items():
            network_type = network_info["type"]
            network_input = network_info["data"]
            network_args = network_info["args"]
            network_args["name"] = network_name
            network = ENCODER_DICT[network_type](**network_args)
            components["network"] = network
            dict_networks[network_name] = {"network": network,
                                           "input": network_input}
            break
        components["dict_networks"] = dict_networks

        # Dataset
        dataset_config = config_dict["dataset"]
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
            if components["logger"] is not None:
                components["logger"]["logger"].log_dataset(
                    dataset_name=dataset_name,
                    pd_df_dict=pd_df_dict)

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

        # Normalizer
        components["normalizer"] = train_set.get_normalizers()

        # Loss function
        loss_func = config_dict["loss_func"]["type"]
        components["loss_func"] = LOSS_FUNC_DICT[loss_func]
        components["loss_func_args"] = config_dict["loss_func"]["args"]

        # TrajectoriesReconstructor
        if config_dict["loss_func"]["type"] == "mp_rec_mse_loss":
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
                                                   batch_size=len(validate_set),
                                                   shuffle=shuffle,
                                                   num_workers=num_workers,
                                                   generator=components["rng"])
        components["test_loader"] = DataLoader(test_set,
                                               batch_size=len(test_set),
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               generator=components["rng"])

        # Data assignment manager
        data_manager_config = config_dict["data_assignment_manager"]
        manager_type = data_manager_config["type"]
        data_manager_args = data_manager_config["args"]
        data_manager_config = {"data_info": train_set.get_data_info(),
                               "encoder_info": dict_networks,
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
        return MPNet.read_local_dataset(dataset_config, random_seed)

    def _get_net_params(self):
        """
        Get parameters to be optimized
        Returns:
             Tuple of parameters of neural networks

        """
        parameters = list()
        network = self._comps["network"]
        parameters += network.parameters

        return (parameters)

    def fit(self):
        """
        External entrance of main training and validation

        Returns:
            None
        """

        # Network tuples
        networks = tuple()

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

        network_input = assigned_dict["dict_encoder_input"]

        reconstructor_input = assigned_dict["reconstructor_input"]
        final_ground_truth = assigned_dict["final_ground_truth"]

        # Predict
        pred = self._predict(network_input=network_input)

        # Compute loss
        loss_kwargs = {"normalizer": self.normalizer,
                       "reconstructor": self.reconstructor,
                       "reconstructor_input": reconstructor_input,
                       "final_ground_truth": final_ground_truth}

        loss = self._comps["loss_func"](true_val=None,
                                        pred_mean=pred,
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
        self._comps["network"].save_weights(log_dir, self.epoch)

    def _load_weights_from_file(self,
                                log_dir):
        """
        Load NN weights from files
        Args:
            log_dir: log directory to write files

        Returns:
            None
        """
        self._comps["network"].load_weights(log_dir, self.epoch)

    def _predict(self, network_input):
        """
        Internal prediction call

        Args:
            network_input: input of the network

        Returns:
            predicted values
        """
        for network_name, network_info in self._comps["dict_networks"].items():
            # Get encoder
            network = network_info["network"]
            # Get data assigned to it
            context_batch = network_input[network_name]
            break

        # NN forward pass
        output = self._comps["network"].encode(context_batch)

        # De-normalize
        predict_key = self._comps["data_manager"].predict_key
        output_dict = \
            BatchProcess.batch_denormalize(
                normalizer=self.normalizer,
                dict_batch={predict_key: {"value": output}})
        pred = output_dict[predict_key]["value"]

        # Return
        return pred

    @torch.no_grad()
    def predict(self,
                dict_obs):
        """
        External prediction call, as a wrapper
        Args:
            dict_obs: dictionary of observed data

        Returns:
            predicted mean, predict covariance diagonal, and off-diagonal
        """
        # Assign data
        data_manager = self._comps["data_manager"]
        data_manager.feed_data_batch(dict_obs, inference_only=True)
        assigned_dict = data_manager.assign_data(inference_only=True)

        network_input = assigned_dict["dict_encoder_input"]

        # Predict
        pred = self._predict(network_input)

        # Return
        return pred

    @torch.no_grad()
    def use_test_dataset_to_predict(self,
                                    **kwargs):
        """
        Use test dataset to make prediction
        Args:

        Returns:
            result_dict: a dictionary containing results for post processing
        """

        # 1. Get test data
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
        network_input = assigned_dict["dict_encoder_input"]

        pred = self._predict(network_input)

        # Get result dictionary
        result_dict = {"test_batch": test_batch,
                       "loss": loss,
                       "assigned_dict": assigned_dict,
                       "pred": pred}

        return result_dict

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
