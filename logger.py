"""
--
@brief:     Log functions
"""
import csv
from nmp import util
import wandb
import os
import pandas as pd
import matplotlib.pyplot as plt


class WandbLogger:
    def __init__(self, config):
        """
        Initialize wandb logger
        Args:
            config: config file of current task
        """
        self.project_name = config["logger"]["log_name"]
        self._initialize_log_dir()
        self._run = wandb.init(project=self.project_name,
                               config=config)

    def _initialize_log_dir(self):
        """
        Clean and initialize local log directory
        Returns:
            True if successfully cleaned
        """
        # Clean old log
        util.remove_file_dir(self.log_dataset_dir)
        util.remove_file_dir(self.log_model_dir)
        util.remove_file_dir(self.log_dir)

        # Make new directory
        os.makedirs(self.log_dir)
        os.makedirs(self.log_dataset_dir)
        os.makedirs(self.log_model_dir)

    @property
    def config(self):
        """
        Log configuration file

        Returns:
            synchronized config from wandb server

        """
        return wandb.config

    @property
    def log_dir(self):
        """
        Get local log saving directory
        Returns:
            log directory
        """
        if not hasattr(self, "_log_dir"):
            self._log_dir = util.get_log_dir(self.project_name)

        return self._log_dir

    @property
    def log_dataset_dir(self):
        """
        Get downloaded logged dataset directory
        Returns:
            logged dataset directory
        """
        return os.path.join(self.log_dir, "dataset")

    @property
    def log_model_dir(self):
        """
        Get downloaded logged model directory
        Returns:
            logged model directory
        """
        return os.path.join(self.log_dir, "model")

    def log_dataset(self,
                    dataset_name,
                    pd_df_dict: dict):
        """
        Log raw dataset to Artifact

        Args:
            dataset_name: Name of dataset
            pd_df_dict: dictionary of train, validate and test sets

        Returns:
            None
        """

        # Initialize wandb Artifact
        raw_data = wandb.Artifact(name=dataset_name + "_dataset",
                                  type="dataset",
                                  description="dataset")

        # Save DataFrames in Artifact
        for key, value in pd_df_dict.items():
            for index, pd_df in enumerate(value):
                with raw_data.new_file(key + "_{}.csv".format(index),
                                       mode="w") as file:
                    file.write(pd_df.to_csv(path_or_buf=None,
                                            index=False,
                                            quoting=csv.QUOTE_ALL))

        # Log Artifact
        self._run.log_artifact(raw_data)

    def load_dataset(self,
                     dataset_config: dict,
                     dataset_api: str):
        """
        Load dataset from Artifact
        # TODO: Check if dataset partition match
        Args:
            dataset_config: dataset_config, used to check if the config and
            logged dataset match
            dataset_api: the string for loading dataset

        Returns:
            dataset_group_dict: dataset group dictionary

        """
        # Download dataset Artifact
        dataset_api = "self._" + dataset_api[11:]
        artifact = eval(dataset_api)
        artifact.download(root=self.log_dataset_dir)
        file_names = util.get_file_names_in_directory(self.log_dataset_dir)
        file_names.sort()

        # Group names
        dataset_group_names = ["train_pd_df",
                               "train_static_pd_df",
                               "validate_pd_df",
                               "validate_static_pd_df",
                               "test_static_pd_df",
                               "test_pd_df"]

        # From file to dataset
        dataset_group_dict = dict()
        for group_name in dataset_group_names:
            group_file_names = [file_name for file_name in file_names if
                                group_name in file_name]
            list_pd_df = list()
            for file_name in group_file_names:
                try:
                    pd_df = pd.read_csv(
                        os.path.join(self.log_dataset_dir, file_name),
                        quoting=csv.QUOTE_ALL)
                except pd.errors.EmptyDataError:
                    pd_df = pd.DataFrame()
                list_pd_df.append(pd_df)
            dataset_group_dict[group_name] = list_pd_df

        # Return
        return dataset_group_dict

    def log_info(self,
                 epoch,
                 key,
                 value):
        self._run.log({"Epoch": epoch,
                       key: value})

    def log_model(self,
                  finished: bool = False):
        """
        Log model into Artifact

        Args:
            finished: True if current training is finished, this will clean
            the old model version without any special aliass

        Returns:
            None
        """
        # Initialize wandb artifact
        model_artifact = wandb.Artifact(name="model", type="model")

        # Get all file names in log dir
        file_names = util.get_file_names_in_directory(self.log_model_dir)

        # Add files into artifact
        for file in file_names:
            path = os.path.join(self.log_model_dir, file)
            model_artifact.add_file(path)

        if finished:
            aliases = ["latest",
                       "finished-{}".format(util.get_formatted_date_time())]
        else:
            aliases = ["latest"]

        # Log and upload
        self._run.log_artifact(model_artifact, aliases=aliases)

        if finished:
            self.delete_useless_model()

    def delete_useless_model(self):
        """
        Delete useless models in WandB server
        Returns:
            None

        """
        api = wandb.Api()

        artifact_type = "model"
        artifact_name = "{}/{}/model".format(self._run.entity,
                                             self._run.project)

        for version in api.artifact_versions(artifact_type, artifact_name):
            # Clean up all versions that don't have an alias such as 'latest'.
            if len(version.aliases) == 0:
                version.delete()

    def load_model(self,
                   model_api: str):
        """
        Load model from Artifact

        model_api: the string for load the model if init_epoch is not zero

        Returns:
            model_dir: Model's directory

        """
        model_api = "self._" + model_api[11:]
        artifact = eval(model_api)
        artifact.download(root=self.log_model_dir)
        file_names = util.get_file_names_in_directory(self.log_model_dir)
        file_names.sort()
        util.print_line_title(title="Download model files from WandB")
        for file in file_names:
            print(file)
        return self.log_model_dir

    def watch_networks(self,
                       networks,
                       log_freq):
        """
        Watch Neural Network weights and gradients
        Args:
            networks: network to being watched
            log_freq: frequency for logging

        Returns:
            None

        """
        for idx, net in enumerate(networks):
            self._run.watch(net,
                            log="all",
                            log_freq=log_freq,
                            idx=idx)

    def log_figure(self,
                   figure_obj: plt.Figure,
                   figure_name: str = "Unnamed Figure"):
        """
        Log figure
        Args:
            figure_obj: Matplotlib Figure object
            figure_name: name of the figure

        Returns:
            None

        """
        self._run.log({figure_name: wandb.Image(figure_obj)})

    def log_video(self,
                  path_to_video: str,
                  video_name: str = "Unnamed Video"):
        """
        Log video
        Args:
            path_to_video: path where the video is stored
            video_name: name of the video

        Returns:
            None
        """
        self._run.log({video_name: wandb.Video(path_to_video)})

    def log_data_dict(self,
                      data_dict: dict):
        """
        Log data in dictionary
        Args:
            data_dict: dictionary to log

        Returns:
            None
        """
        self._run.log(data_dict)


def get_logger_dict():
    return {"wandb": WandbLogger}