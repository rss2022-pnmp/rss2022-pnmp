"""
--
@brief:     Classes of Neural Network Bases
"""
import pickle as pkl
import torch
from torch import nn as nn
from torch.nn import ModuleList
from torch.nn import functional as F
from nmp import util


class MLP(nn.Module):
    def __init__(self,
                 name: str = None,
                 log_dir: str = None,
                 load_epoch: int = None,
                 dim_in: int = None,
                 dim_out: int = None,
                 hidden_layers: list = None,
                 act_func: str = None,
                 seed: int = None):
        """
        Multi-layer Perceptron Constructor

        Args:
            name: name of the MLP
            log_dir: load weights file if not None
            load_epoch: which epoch to be loaded from file if not None
            dim_in: dimension of input
            dim_out: dimension of output
            hidden_layers: a list containing hidden layers' dimensions
            act_func: activation function
            seed: seed for random behaviours
        """

        super(MLP, self).__init__()

        self.act_func_dict = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "leaky_relu": F.leaky_relu,
            "softplus": F.softplus,
            "exp": torch.exp,
            "None": None,
        }

        self.name = name
        self.mlp_name = name + "_mlp"

        # Initialize the MLP
        if log_dir is None:
            # Initialize new net
            assert load_epoch is None
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.hidden_layers = hidden_layers
            self.act_func = self.act_func_dict[act_func]
            self.seed = seed

            # Create networks
            # Ugly but useful to distinguish networks in gradient watch
            # e.g. if self.mlp_name is "encoder_mlp"
            # Then below will lead to self.encoder_mlp = self.__create_network()
            setattr(self, self.mlp_name, self.__create_network())

        else:
            # Initialize and create network from file
            self.load(log_dir, load_epoch)

    def __create_network(self):
        """
        Create MLP Network

        Returns:
        MLP Network
        """
        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Total layers (n+1) = hidden layers (n) + output layer (1)

        # Add first hidden layer
        mlp = ModuleList([nn.Linear(in_features=self.dim_in,
                                    out_features=self.hidden_layers[0])])

        # Add other hidden layers
        for i in range(1, len(self.hidden_layers)):
            mlp.append(nn.Linear(in_features=mlp[-1].out_features,
                                 out_features=self.hidden_layers[i]))

        # Add output layer
        mlp.append(nn.Linear(in_features=mlp[-1].out_features,
                             out_features=self.dim_out))

        return mlp

    def save(self, log_dir, epoch):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "hidden_layers": self.hidden_layers,
                "act_func": self.act_func.__name__ if self.act_func is not None
                else "None",
                "seed": self.seed,
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir, epoch):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.name, epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            self.dim_in = parameters["dim_in"]
            self.dim_out = parameters["dim_out"]
            self.hidden_layers = parameters["hidden_layers"]
            self.act_func = self.act_func_dict[parameters["act_func"]]
            self.seed = parameters["seed"]

        # Create network
        setattr(self, self.mlp_name, self.__create_network())

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, input_data):
        """
        Network forward function

        Args:
            input_data: input data

        Returns: MLP output

        """
        data = input_data

        # Hidden layers (n) + output layer (1)
        mlp = eval("self." + self.mlp_name)
        for i in range(len(self.hidden_layers)):
            data = self.act_func(mlp[i](data))
        data = mlp[-1](data)

        # Return
        return data


# class MNISTNet(nn.Module):
#     """
#     Network for Mnist
#     """
#
#     def __init__(self):
#         super(MNISTNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


class DigitCNN(nn.Module):
    def __init__(self,
                 name: str = None,
                 log_dir: str = None,
                 load_epoch: int = None,
                 dim_out: int = None,
                 hidden_layers: list = None,
                 act_func: str = None,
                 seed: int = None,
                 image_size: int = None):

        super(DigitCNN, self).__init__()
        self.act_func_dict = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "leaky_relu": F.leaky_relu,
            "softplus": F.softplus,
            "exp": torch.exp,
            "None": None,
        }

        self.name = name + "_digit"
        self.cnn_mlp_name = name + "_cnn_mlp"
        self.image_size = image_size if image_size is not None else 40

        # Get net
        if log_dir is None:
            # Initialize new net
            assert load_epoch is None
            self.dim_out = dim_out
            self.hidden_layers = hidden_layers
            self.act_func = self.act_func_dict[act_func]
            self.seed = seed

            # Create a new net
            setattr(self, self.cnn_mlp_name, self.__create_network())

        else:
            # Initialize and create network from file
            self.load(log_dir, load_epoch)

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=1):
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        return (size - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def maxpool2d_size_out(size, kernel_size=2, stride=None):
        if stride is None:
            stride = kernel_size
        return DigitCNN.conv2d_size_out(size, kernel_size=kernel_size,
                                        stride=stride)

    def __create_network(self):
        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Set CNN
        conv1 = nn.Conv2d(1, 10, kernel_size=5)
        conv2 = nn.Conv2d(10, 20, kernel_size=5)
        cnn_mlp = ModuleList([conv1, conv2])

        image_out_size = \
            self.maxpool2d_size_out(
                self.conv2d_size_out(
                    self.maxpool2d_size_out(
                        self.conv2d_size_out(self.image_size))))
        self.linear_input_size = image_out_size * image_out_size * 20

        # Add first hidden layer
        cnn_mlp.append(nn.Linear(in_features=self.linear_input_size,
                                 out_features=self.hidden_layers[0]))
        # Add other hidden layers
        for i in range(1, len(self.hidden_layers)):
            cnn_mlp.append(nn.Linear(in_features=cnn_mlp[-1].out_features,
                                     out_features=self.hidden_layers[i]))
        # Add output layer
        cnn_mlp.append(nn.Linear(in_features=cnn_mlp[-1].out_features,
                                 out_features=self.dim_out))
        return cnn_mlp

    def save(self, log_dir, epoch):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "dim_out": self.dim_out,
                "hidden_layers": self.hidden_layers,
                "act_func": self.act_func.__name__ if self.act_func is not None
                else "None",
                "seed": self.seed,
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir, epoch):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.name, epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            self.dim_out = parameters["dim_out"]
            self.hidden_layers = parameters["hidden_layers"]
            self.act_func = self.act_func_dict[parameters["act_func"]]
            self.seed = parameters["seed"]

        # Create network
        setattr(self, self.cnn_mlp_name, self.__create_network())

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, input_data):

        # Reshape images batch to [num_traj * num_obs, C=1, H, W]
        num_traj, num_obs = input_data.shape[:2]
        input_data = input_data.reshape(-1, *input_data.shape[2:])

        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        cnn_mlp = eval("self." + self.cnn_mlp_name)

        # CNN layer (2)
        data = self.act_func(F.max_pool2d(cnn_mlp[0](input_data), 2))
        data = self.act_func(F.max_pool2d(
            F.dropout2d(cnn_mlp[1](data), training=self.training), 2))

        # Flatten
        data = data.view(num_traj, num_obs, self.linear_input_size)

        # MLP layers (n) + output layer (1)
        for i in range(2, len(cnn_mlp) - 1):
            data = self.act_func(cnn_mlp[i](data))
        data = cnn_mlp[-1](data)

        # Return
        return data


class GRURNN(nn.Module):
    def __init__(self,
                 name: str = None,
                 log_dir: str = None,
                 load_epoch: int = None,
                 dim_in: int = None,
                 dim_out: int = None,
                 num_layers: int = None,
                 seed: int = None):
        """
        Gated Recurrent Unit of RNN

        Args:
            name: name of the GRU
            log_dir: load weights file if not None
            load_epoch: which epoch to be loaded from file if not None
            dim_in: dimension of input
            dim_out: dimension of output
            num_layers: number of hidden layers
            seed: seed for random behaviours
        """

        super(GRURNN, self).__init__()

        self.name = name
        self.gru_name = name + "_gru"

        # Initialize the GRU
        if log_dir is None:
            # Initialize new net
            assert load_epoch is None
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.num_layers = num_layers
            self.seed = seed

            # Create networks
            setattr(self, self.gru_name, self.__create_network())

        else:
            # Initialize and create network from file
            self.load(log_dir, load_epoch)

    def __create_network(self):
        """
        Create GRU Network

        Returns:
        GRU Network
        """
        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Add gru
        gru = nn.GRU(input_size=self.dim_in,
                     hidden_size=self.dim_out,
                     num_layers=self.num_layers,
                     batch_first=True)

        return gru

    def save(self, log_dir, epoch):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """

        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "num_layers": self.num_layers,
                "seed": self.seed,
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir, epoch):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.name, epoch)

        # Load structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            self.dim_in = parameters["dim_in"]
            self.dim_out = parameters["dim_out"]
            self.num_layers = parameters["num_layers"]
            self.seed = parameters["seed"]

        # Create network
        setattr(self, self.gru_name, self.__create_network())

        # Load NN weights
        self.load_state_dict(torch.load(w_path))

    def forward(self, input_data):
        """
        Network forward function

        Args:
            input_data: input data

        Returns: GRU output

        """
        data = input_data

        gru = eval("self." + self.gru_name)
        data = gru(data)

        # Return
        return data
