"""
--
@brief:     Decoder classes in PyTorch
"""

# Import Python libs
import torch
from nmp.nn_base import MLP
from nmp.util import de_param
from nmp.util import var_param


class Decoder:
    """Decoder class interface"""

    def __init__(self, **kwargs):
        """
        Constructor

        Args:
            **kwargs: Decoder configuration
        """

        # MLP configuration
        self.dim_time: int = kwargs["dim_time"]
        self.dim_cond: int = kwargs.get("dim_cond", 0)
        self.dim_val: int = kwargs["dim_val"]
        self.dim_val_cov: int = kwargs.get("dim_val_cov", self.dim_val)
        self.dim_lat: int = kwargs["dim_lat"]
        self.std_only: bool = kwargs["std_only"]

        self.hidden_layers_mean_val: list = kwargs["hidden_layers_mean_val"]
        self.hidden_layers_cov_val: list = kwargs["hidden_layers_cov_val"]

        self.act_func: str = kwargs["act_func"]
        self.seed: int = kwargs["seed"]

        # Decoders
        self.mean_val_net = None
        self.cov_val_net = None

        # Create decoders
        self._create_network()

    @property
    def _decoder_type(self) -> str:
        """
        Returns: string of decoder type
        """
        return self.__class__.__name__

    def _create_network(self):
        """
        Create decoder with given configuration

        Returns:
            None
        """

        # compute the output dimension of covariance network
        if self.std_only:
            # Only has diagonal elements
            dim_out_cov = self.dim_val_cov
        else:
            # Diagonal + Non-diagonal elements, form up Cholesky Decomposition
            dim_out_cov = self.dim_val_cov \
                          + (self.dim_val_cov * (self.dim_val_cov - 1)) // 2

        # Two separate value decoders: mean_val_net + cov_val_net

        self.mean_val_net = MLP(name=self._decoder_type + "_mean_val",
                                dim_in=self.dim_time + self.dim_cond +
                                       self.dim_lat,
                                dim_out=self.dim_val,
                                hidden_layers=self.hidden_layers_mean_val,
                                act_func=self.act_func,
                                seed=self.seed)

        self.cov_val_net = MLP(name=self._decoder_type + "_cov_val",
                               dim_in=self.dim_time + self.dim_cond +
                                      self.dim_lat,
                               dim_out=dim_out_cov,
                               hidden_layers=self.hidden_layers_cov_val,
                               act_func=self.act_func,
                               seed=self.seed)

    @property
    def network(self):
        """
        Return decoder networks

        Returns:
        """
        return self.mean_val_net, self.cov_val_net

    @property
    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.mean_val_net.parameters()) + \
               list(self.cov_val_net.parameters())

    def save_weights(self, log_dir, epoch):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.mean_val_net.save(log_dir, epoch)
        self.cov_val_net.save(log_dir, epoch)

    def load_weights(self, log_dir, epoch):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.mean_val_net.load(log_dir, epoch)
        self.cov_val_net.load(log_dir, epoch)

    def _process_cov_net_output(self, cov_val):
        """
        Divide diagonal and off-diagonal elements of cov-net output,
        apply reverse "Log-Cholesky to diagonal elements"
        Args:
            cov_val: output of covariance network

        Returns: diagonal and off-diagonal tensors

        """
        # Decompose diagonal and off-diagonal elements
        diag_cov_val = cov_val[..., :self.dim_val_cov]
        off_diag_cov_val = None if self.std_only \
            else cov_val[..., self.dim_val_cov:]

        # De-parametrize Log-Cholesky for diagonal elements
        diag_cov_val = de_param(diag_cov_val)

        # Return
        return diag_cov_val, off_diag_cov_val


class PBDecoder(Decoder):
    """Parameter based decoder"""

    def decode(self,
               start_value,
               times,
               mean_lat_var,
               variance_lat_var):
        """
        Decode and compute target value's distribution at target times

        Here, target value to be predicted is a 4th order tensor with axes:

        traj: this target value is on which trajectory
        aggr: based on how much aggregated context do we make this prediction?
        tar: this target value is on which target time?
        value: vector to be predicted

        Args:
            start_value: start values, can be None
            times: query target times, can be None
            mean_lat_var: mean of latent variable
            variance_lat_var: variance of latent variable

        Returns:
            mean_val: mean of target value

            diag_cov_val: diagonal elements of Cholesky Decomposition of
            covariance of target value

            off_diag_cov_val: None, or off-diagonal elements of Cholesky
            Decomposition of covariance of target value

        """

        # Shape of mean_lat_var:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of variance_lat_var:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of times:
        # [num_traj, num_time_pts, dim_time=1]
        #
        # Shape of start_value:
        # [num_traj, 1, dim_cond]
        #
        # Shape of mean_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of diag_cov_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of off_diag_cov_val:
        # [num_traj, num_agg, num_time_pts, (dim_val * (dim_val - 1) // 2)]

        # Dimension check
        assert mean_lat_var.ndim == variance_lat_var.ndim == 3
        num_agg = mean_lat_var.shape[1]

        # Process times
        if times is not None:
            assert times.ndim == 3
            num_time_pts = times.shape[1]
            # Add one axis (aggregation-wise batch dimension) to times
            times = times[:, None, :, :]
            times = times.expand(-1, num_agg, -1, -1)
        else:
            num_time_pts = 1

        # Parametrize variance
        variance_lat_var = var_param(variance_lat_var)

        # Add one axis (time-scale-wise batch dimension) to latent variable
        mean_lat_var = mean_lat_var[:, :, None, :]
        variance_lat_var = variance_lat_var[:, :, None, :]
        mean_lat_var = mean_lat_var.expand(-1, -1, num_time_pts, -1)
        variance_lat_var = variance_lat_var.expand(-1, -1, num_time_pts, -1)

        # Prepare
        if start_value is not None:
            start_value = start_value[:, None, :, :]
            start_value = start_value.expand(-1, num_agg, num_time_pts, -1)

        # Prepare input to decoder networks
        mean_net_input = mean_lat_var
        cov_net_input = variance_lat_var
        if times is not None:
            mean_net_input = torch.cat((times, mean_net_input), dim=-1)
            cov_net_input = torch.cat((times, cov_net_input), dim=-1)
        if start_value is not None:
            mean_net_input = torch.cat((start_value, mean_net_input), dim=-1)
            cov_net_input = torch.cat((start_value, cov_net_input), dim=-1)

        # Decode
        mean_val = self.mean_val_net(mean_net_input)
        cov_val = self.cov_val_net(cov_net_input)

        # Process cov net prediction
        diag_cov_val, off_diag_cov_val = self._process_cov_net_output(cov_val)

        # Return
        return mean_val, diag_cov_val, off_diag_cov_val


class CNPDecoder(Decoder):
    """Conditional Neural Processes decoder"""

    def decode(self,
               start_value,
               times,
               mean_lat_obs):
        """
        Decode and compute target value's distribution at target times

        Here, target value to be predicted is a 4rd order tensor with axes:

        traj: this target value is on which trajectory
        aggr: based on how much aggregated context do we make this prediction?
        tar: this target value is on which target time?
        value: vector to be predicted

        Args:
            start_value: start values, can be None
            times: query target times, can be None
            mean_lat_obs: mean of latent observation

        Returns:
            mean_val: mean of target value

            diag_cov_val: diagonal elements of Cholesky Decomposition of
            covariance of target value

            off_diag_cov_val: None, or off-diagonal elements of Cholesky
            Decomposition of covariance of target value

        """

        # Shape of mean_lat_obs:
        # [num_traj, num_agg, dim_lat]
        #
        # Shape of times:
        # [num_traj, num_time_pts, dim_time=1] if times not None
        #
        # Shape of start_value:
        # [num_traj, 1, dim_cond]
        #
        # Shape of mean_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of diag_cov_val:
        # [num_traj, num_agg, num_time_pts, dim_val]
        #
        # Shape of off_diag_cov_val:
        # [num_traj, num_agg, num_time_pts, (dim_val * (dim_val - 1) // 2)]

        # Dimension check
        assert mean_lat_obs.ndim == 3
        num_agg = mean_lat_obs.shape[1]

        # Process times
        if times is not None:
            assert times.ndim == 3
            # Get dimensions
            num_time_pts = times.shape[1]
            # Add one axis (aggregation-wise batch dimension) to times
            times = times[:, None, :, :]
            times = times.expand(-1, num_agg, -1, -1)
        else:
            num_time_pts = 1

        # Add one axis (time-scale-wise batch dimension) to latent observation
        mean_lat_obs = mean_lat_obs[:, :, None, :]
        mean_lat_obs = mean_lat_obs.expand(-1, -1, num_time_pts, -1)

        # Prepare
        # TODO REFACTOR
        if start_value is not None:
            start_value = start_value[:, None, :, :]
            start_value = start_value.expand(-1, num_agg, num_time_pts, -1)

        # Prepare input to decoder network
        net_input = mean_lat_obs
        if times is not None:
            net_input = torch.cat((times, net_input), dim=-1)
        if start_value is not None:
            net_input = torch.cat((start_value, net_input), dim=-1)

        # Decode
        mean_val = self.mean_val_net(net_input)
        cov_val = self.cov_val_net(net_input)

        # Process cov net prediction
        diag_cov_val, off_diag_cov_val = self._process_cov_net_output(cov_val)

        # Return
        return mean_val, diag_cov_val, off_diag_cov_val


class MCDecoder(Decoder):
    """Monte-Carlo decoder"""

    def decode(self,
               start_value,
               times,
               sampled_lat_var,
               variance_lat_var):
        """
        Decode and compute target value's distribution at target times

        Here, target value to be predicted is a 5th order tensor with axes:

        traj: this target value is on which trajectory
        aggr: based on how much aggregated context do we make this prediction?
        sample: latent variable samples for Monte-Carlo
        tar: this target value is on which target time?
        value: vector to be predicted

        Args:
            start_value: start values, can be None
            times: query target times, can be None
            sampled_lat_var: sampled latent variable
            variance_lat_var: variance of latent variable

        Returns:
            mean_val: mean of target value

            diag_cov_val: diagonal elements of Cholesky Decomposition of
            covariance of target value

            off_diag_cov_val: None, or off-diagonal elements of Cholesky
            Decomposition of covariance of target value
        """

        # Shape of sampled_lat_var:
        # [num_traj, num_agg, num_smp, dim_lat]
        #
        # Shape of variance_lat_var:
        # [num_traj, num_agg, num_smp, dim_lat]
        #
        # Shape of times:
        # [num_traj, num_time_pts, dim_time=1] if times not None
        #
        # Shape of start_value:
        # [num_traj, 1, dim_cond]
        #
        # Shape of mean_val:
        # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
        #
        # Shape of diag_cov_val:
        # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
        #
        # Shape of off_diag_cov_val:
        # [num_traj, num_agg, num_smp, num_time_pts,
        # (dim_val * (dim_val - 1) // 2)]

        # Dimension check
        assert sampled_lat_var.ndim == variance_lat_var.ndim == 4
        num_agg = sampled_lat_var.shape[1]
        num_smp = sampled_lat_var.shape[2]

        # Process times
        if times is not None:
            assert times.ndim == 3
            # Get dimensions
            num_time_pts = times.shape[1]
            # Add one axis (aggregation-wise batch dimension) to times
            times = times[:, None, None, ...]
            times = times.expand(-1, num_agg, num_smp, -1, -1)
        else:
            num_time_pts = 1

        # Parametrize variance
        variance_lat_var = var_param(variance_lat_var)

        # Add one axis (time-scale-wise batch dimension) to latent observation
        sampled_lat_var = sampled_lat_var[..., None, :]
        sampled_lat_var = sampled_lat_var.expand(-1, -1, -1, num_time_pts, -1)
        variance_lat_var = variance_lat_var[..., None, :]
        variance_lat_var = variance_lat_var.expand(-1, -1, -1, num_time_pts, -1)

        # Prepare
        if start_value is not None:
            start_value = start_value[:, None, None, :, :]
            start_value = start_value.expand(-1, num_agg, num_smp, num_time_pts,
                                             -1)

        # Prepare input to decoder network
        mean_net_input = sampled_lat_var
        cov_net_input = variance_lat_var
        if times is not None:
            mean_net_input = torch.cat((times, sampled_lat_var), dim=-1)
            cov_net_input = torch.cat((times, variance_lat_var), dim=-1)
        if start_value is not None:
            mean_net_input = torch.cat((start_value, mean_net_input), dim=-1)
            cov_net_input = torch.cat((start_value, cov_net_input), dim=-1)

        # Decode
        mean_val = self.mean_val_net(mean_net_input)
        cov_val = self.cov_val_net(cov_net_input)

        # Process cov net prediction
        diag_cov_val, off_diag_cov_val = self._process_cov_net_output(cov_val)

        # Return
        return mean_val, diag_cov_val, off_diag_cov_val


def get_decoder_dict():
    return {"PBDecoder": PBDecoder,
             "CNPDecoder": CNPDecoder,
             "MCDecoder": MCDecoder}
