"""
--
@brief:     Encoder classes in PyTorch
"""

# Import Python libs
import torch

from nmp.nn_base import MLP, GRURNN, DigitCNN
from nmp.util import de_param

class NMPEncoder:

    def __init__(self, **kwargs):
        """
        NMP encoder constructor
        Args:
            **kwargs: Encoder configuration
        """

        # MLP configuration
        self.name: str = kwargs["name"]
        self.dim_time: int = kwargs["dim_time"]
        self.dim_val: int = kwargs["dim_val"]
        self.dim_lat_obs: int = kwargs["dim_lat"]

        self.hidden_layers_obs: list = kwargs["hidden_layers_obs"]
        self.hidden_layers_unc_obs: list = kwargs["hidden_layers_unc_obs"]

        self.act_func: str = kwargs["act_func"]
        self.seed: int = kwargs["seed"]

        # Encoders
        self.lat_obs_net = None
        self.var_lat_obs_net = None

        # Create latent observation encoder
        self.__create_network()

    def __create_network(self):
        """
        Create encoder with given configuration

        Returns:
            None
        """
        # Two separate latent observation encoders
        # lat_obs_net + var_lat_obs_net
        self.lat_obs_net = MLP(name="NMPEncoder_lat_obs_" + self.name,
                               dim_in=self.dim_time + self.dim_val,
                               dim_out=self.dim_lat_obs,
                               hidden_layers=self.hidden_layers_obs,
                               act_func=self.act_func,
                               seed=self.seed)

        self.var_lat_obs_net = MLP(
            name="NMPEncoder_var_lat_obs_" + self.name,
            dim_in=self.dim_time + self.dim_val,
            dim_out=self.dim_lat_obs,
            hidden_layers=self.hidden_layers_unc_obs,
            act_func=self.act_func,
            seed=self.seed)

    @property
    def network(self):
        """
        Return encoder networks

        Returns:
        """
        return self.lat_obs_net, self.var_lat_obs_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.lat_obs_net.parameters()) + \
               list(self.var_lat_obs_net.parameters())

    def save_weights(self, log_dir, epoch):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.save(log_dir, epoch)
        self.var_lat_obs_net.save(log_dir, epoch)

    def load_weights(self, log_dir, epoch):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.load(log_dir, epoch)
        self.var_lat_obs_net.load(log_dir, epoch)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
            var_lat_obs: variance of latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, dim_time + dim_val],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 3

        # Encode
        return self.lat_obs_net(obs), \
               de_param(self.var_lat_obs_net(obs))

    # End of class NMPEncoder


class CNMPEncoder:

    def __init__(self, **kwargs):
        """
        CNMP encoder constructor

        Args:
            **kwargs: Encoder configuration
        """

        # MLP configuration
        self.name: str = kwargs["name"]
        self.dim_time: int = kwargs["dim_time"]
        self.dim_val: int = kwargs["dim_val"]
        self.dim_lat_obs: int = kwargs["dim_lat"]

        self.hidden_layers_obs: list = kwargs["hidden_layers_obs"]
        self.act_func: str = kwargs["act_func"]
        self.seed: int = kwargs["seed"]

        # Encoder architecture
        self.lat_obs_net = None

        # Latent observation encoder (lat_obs_net)
        self.__create_network()

    def __create_network(self):
        """
        Create encoder network with given configuration

        Returns:
            None
        """
        self.lat_obs_net = MLP(name="CNMPEncoder_lat_obs_" + self.name,
                               dim_in=self.dim_time + self.dim_val,
                               dim_out=self.dim_lat_obs,
                               hidden_layers=self.hidden_layers_obs,
                               act_func=self.act_func,
                               seed=self.seed)

    @property
    def network(self):
        """
        Return encoder network

        Returns:
        """
        return self.lat_obs_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """

        return list(self.lat_obs_net.parameters())

    def save_weights(self, log_dir, epoch):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        self.lat_obs_net.save(log_dir, epoch)

    def load_weights(self, log_dir, epoch):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.load(log_dir, epoch)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, dim_time + dim_val],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 3

        # Encode
        return self.lat_obs_net(obs)


class GruMlpNet:
    def __init__(self, **kwargs):
        """
        Recurrent encoder constructor

        Args:
            **kwargs: Encoder configuration
        """

        # NN configuration
        self.name: str = kwargs["name"]
        self.dim_time: int = kwargs["dim_time"]
        self.dim_val: int = kwargs["dim_val"]
        self.dim_out: int = kwargs["dim_out"]

        # GRU layer settings
        self.gru_out = kwargs["gru_out"]
        self.num_gru_layers = kwargs["num_gru_layers"]

        # FC layer setttings
        self.hidden_layers: list = kwargs["hidden_layers"]
        self.act_func: str = kwargs["act_func"]

        self.seed: int = kwargs["seed"]

        # Encoder architecture
        self.gru_net = None
        self.mlp_net = None

        # Latent observation encoder (lat_obs_net)
        self.__create_network()

    def __create_network(self):
        """
        Create network with given configuration

        Returns:
            None
        """
        self.gru_net = GRURNN(name="GRU" + self.name,
                              dim_in=self.dim_time + self.dim_val,
                              dim_out=self.gru_out,
                              num_layers=self.num_gru_layers,
                              seed=self.seed)
        self.mlp_net = MLP(name="MLP" + self.name,
                           dim_in=self.gru_out,
                           dim_out=self.dim_out,
                           hidden_layers=self.hidden_layers,
                           act_func=self.act_func,
                           seed=self.seed)

    @property
    def network(self):
        """
        Return encoder network

        Returns:
        """
        return self.gru_net, self.mlp_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.gru_net.parameters()) + \
               list(self.mlp_net.parameters())

    def save_weights(self, log_dir, epoch):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.gru_net.save(log_dir, epoch)
        self.mlp_net.save(log_dir, epoch)

    def load_weights(self, log_dir, epoch):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.gru_net.load(log_dir, epoch)
        self.mlp_net.load(log_dir, epoch)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            output: network output
        """

        # Shape of obs:
        # [num_traj, num_obs, dim_time + dim_val],
        #
        # Shape of output:
        # [num_traj, num_obs, dim_out]
        assert obs.ndim == 3

        result = self.mlp_net(self.gru_net(obs)[0])
        result = result[:, :, None, :]
        return result


class ImageEncoder:
    def __init__(self, **kwargs):
        # configuration
        self.name: str = kwargs["name"]
        self.dim_lat_obs: int = kwargs["dim_lat"]

        self.hidden_layers_obs: list = kwargs["hidden_layers_obs"]
        self.hidden_layers_unc_obs: list = kwargs["hidden_layers_unc_obs"]

        self.image_size = kwargs.get("image_size", 40)

        self.act_func: str = kwargs["act_func"]
        self.seed: int = kwargs["seed"]



        # Encoders
        self.lat_obs_net = None
        self.var_lat_obs_net = None

        # Create latent observation encoder
        self.__create_network()

    def __create_network(self):
        """
        Create encoder with given configuration

        Returns:
            None
        """
        # Two separate latent observation encoders
        # lat_obs_net + var_lat_obs_net
        self.lat_obs_net = DigitCNN(name="NMPEncoder_lat_obs_" + self.name,
                               dim_out=self.dim_lat_obs,
                               hidden_layers=self.hidden_layers_obs,
                               image_size=self.image_size,
                               act_func=self.act_func,
                               seed=self.seed)

        self.var_lat_obs_net = DigitCNN(
            name="NMPEncoder_var_lat_obs_" + self.name,
            dim_out=self.dim_lat_obs,
            hidden_layers=self.hidden_layers_unc_obs,
            image_size=self.image_size,
            act_func=self.act_func,
            seed=self.seed)

    @property
    def network(self):
        """
        Return encoder networks

        Returns:
        """
        return self.lat_obs_net, self.var_lat_obs_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.lat_obs_net.parameters()) + \
               list(self.var_lat_obs_net.parameters())

    def save_weights(self, log_dir, epoch):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.save(log_dir, epoch)
        self.var_lat_obs_net.save(log_dir, epoch)

    def load_weights(self, log_dir, epoch):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.load(log_dir, epoch)
        self.var_lat_obs_net.load(log_dir, epoch)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
            var_lat_obs: variance of latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, C=1, H, W],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 5

        # Encode
        return self.lat_obs_net(obs), \
               de_param(self.var_lat_obs_net(obs))


class CNMPImageEncoder:
    def __init__(self, **kwargs):
        # configuration
        self.name: str = kwargs["name"]
        self.dim_lat_obs: int = kwargs["dim_lat"]

        self.hidden_layers_obs: list = kwargs["hidden_layers_obs"]

        self.image_size = kwargs.get("image_size", 40)

        self.act_func: str = kwargs["act_func"]
        self.seed: int = kwargs["seed"]

        # Encoders
        self.lat_obs_net = None

        # Create latent observation encoder
        self.__create_network()

    def __create_network(self):
        """
        Create encoder with given configuration

        Returns:
            None
        """
        # Two separate latent observation encoders
        # lat_obs_net + var_lat_obs_net
        self.lat_obs_net = DigitCNN(name="CNMPEncoder_lat_obs_" + self.name,
                                    dim_out=self.dim_lat_obs,
                                    hidden_layers=self.hidden_layers_obs,
                                    image_size=self.image_size,
                                    act_func=self.act_func,
                                    seed=self.seed)

    @property
    def network(self):
        """
        Return encoder networks

        Returns:
        """
        return self.lat_obs_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.lat_obs_net.parameters())

    def save_weights(self, log_dir, epoch):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.save(log_dir, epoch)

    def load_weights(self, log_dir, epoch):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.lat_obs_net.load(log_dir, epoch)

    def encode(self, obs):
        """
        Encode observations

        Args:
            obs: observations

        Returns:
            lat_obs: latent observations
            var_lat_obs: variance of latent observations
        """

        # Shape of obs:
        # [num_traj, num_obs, C=1, H, W],
        #
        # Shape of lat_obs:
        # [num_traj, num_obs, dim_lat]
        #
        # Shape of var_lat_obs:
        # [num_traj, num_obs, dim_lat]

        # Check input shapes
        assert obs.ndim == 5

        # Encode
        return self.lat_obs_net(obs)


def get_encoder_dict():
    return {"NMPEncoder": NMPEncoder,
            "CNMPEncoder": CNMPEncoder,
            "GruMlpNet": GruMlpNet,
            "ImageEncoder": ImageEncoder,
            "CNMPImageEncoder": CNMPImageEncoder}
