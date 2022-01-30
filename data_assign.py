"""
    1. CNMP, NMP
    2. CNMP, NMP + PROMP
    3. CNMP, NMP + IDAMP
    4.
    TODO: ADD SOME COMMENTS
"""
import math
import torch

from nmp import util
from nmp.data_process import *


class DataAssignmentInterface:
    """Interface for different data assignment strategies"""

    def __init__(self, **kwargs):
        self._data_info = kwargs["data_info"]
        self._encoder_info = kwargs["encoder_info"]
        self._reconstructor_info = kwargs["reconstructor_info"]
        self._loss_info = kwargs["loss_info"]
        self._normalizer = kwargs["normalizer"]
        self._rng = kwargs["rng"]

        self._predict_key = self._assert_one_predict_key()
        self._traj_key = self._assert_one_traj_key()

        self._num_all = None
        self._data_dict_batch = None
        self._assigned = None
        self._inference_only = False

    @property
    def traj_key(self):
        return self._traj_key

    @property
    def predict_key(self):
        return self._predict_key

    def feed_data_batch(self, data_dict_batch, inference_only=False):
        self._data_dict_batch = data_dict_batch
        if not inference_only:
            self._num_all = self._get_num_all()

    def assign_data(self,
                    inference_only=False,
                    **kwargs):
        self._inference_only = inference_only
        self._assigned = dict()
        for data_name in self._data_dict_batch.keys():
            self._assigned[data_name] = False

        data_dict = dict()
        encoder_input, original_encoder_input = self._get_encoder_input()
        data_dict["dict_encoder_input"] = encoder_input
        data_dict["original_encoder_input"] = original_encoder_input

        norm_decoder_input, original_decoder_input = self._get_decoder_input()
        data_dict["decoder_input"] = norm_decoder_input
        data_dict["original_decoder_input"] = original_decoder_input

        data_dict["decoder_output_ground_truth"] \
            = self._get_decoder_output_ground_truth()
        data_dict["reconstructor_input"] = self._get_reconstructor_input()
        data_dict["final_ground_truth"] = self._get_final_ground_truth()
        return data_dict

    def _assert_one_predict_key(self):
        counter = 0
        predict_key = None
        for key, info in self._data_info.items():
            if info["predict"] is True:
                counter += 1
                predict_key = key
        assert predict_key is not None and counter == 1, \
            "More than 1 kinds of data to be predicted"
        return predict_key

    def _assert_one_traj_key(self):
        counter = 0
        traj_key = None
        for key, info in self._data_info.items():
            if "traj" in info.keys() and info["traj"] is True:
                counter += 1
                traj_key = key
        assert traj_key is not None and counter == 1, \
            "More than 1 trajectory ground-truth"
        return traj_key

    def _get_num_all(self):
        return self._data_dict_batch[self._traj_key]["time"].shape[1]

    def _normalize_data(self, key, data):
        assert data is not None
        if self._normalizer[key] is not None:
            return BatchProcess.batch_normalize({key: data},
                                                self._normalizer)[key]
        else:
            return data

    def _get_encoder_input(self):
        """
        Get encoder input (context), normalized
        Returns:
            dict_encoder_input: dictionary of encoder input
        """
        raise NotImplementedError

    def _get_decoder_input(self):
        """
        Get decoder input, normalized
        Returns:
            Input of the decoder, such as query time points
        """
        raise NotImplementedError

    def _get_decoder_output_ground_truth(self):
        """
        Ground truth of the decoder prediction
        Returns:
            Ground truth of decoder prediction
        """
        raise NotImplementedError

    def _get_reconstructor_input(self):
        """
        Input of trajectory reconstruction, such as time points
        Returns:
            Input for reconstruction
        """
        raise NotImplementedError

    def _get_final_ground_truth(self):
        """
        Get ground truth after post-processing the decoder prediction
        Returns:
            Final ground truth
        """
        raise NotImplementedError


class CtxPredSplitAssignment(DataAssignmentInterface):

    def __init__(self, **kwargs):
        """
        Constructor
        Args:
            **kwargs: keyword arguments
        """
        super(CtxPredSplitAssignment, self).__init__(**kwargs)
        self._check_data_info()

        # Arguments for splitting dataset
        self._split_info = kwargs["split_info"]

        self._ctx_idx, self._pred_idx = None, None

    def assign_data(self,
                    inference_only=False,
                    **kwargs):
        if not inference_only:
            self._ctx_idx, self._pred_idx = \
                self.ctx_pred_index(self._split_info,
                                    self._num_all,
                                    self._rng)
        return super(CtxPredSplitAssignment,
                     self).assign_data(inference_only, **kwargs)

    def _check_data_info(self):
        # Assert there is one kind of data playing both context & predict roles
        assert self._data_info[self._predict_key]["context"] is True

        # Assert prediction of raw trajectory
        assert self._traj_key == self._predict_key

    def _split_ctx(self, data_dict, is_traj=False):
        """
        Split context data from all data, optionally split and save predict data
        too. If inference only, all data will be context data
        Args:
            data_dict: data to be split
            is_traj: flag indicates if both context and predict

        Returns:
            context data
        """
        time_batch = data_dict["time"]
        value_batch = data_dict["value"]
        ctx = dict()
        if not self._inference_only:
            ctx["time"] = time_batch[:, self._ctx_idx]
            ctx["value"] = value_batch[:, self._ctx_idx]
            if is_traj:
                self._decoder_input = time_batch[:, self._pred_idx]
                self._decoder_output_ground_truth = value_batch[:,
                                                    self._pred_idx]
        else:
            ctx["time"] = time_batch
            ctx["value"] = value_batch
        return ctx

    @staticmethod
    def ctx_pred_index(split_info, num_all, rng):
        """
        Compute context and predict index for splitting
        Returns:
            None
        """
        assert split_info is not None, "Need context split info"
        shuffle = split_info["shuffle"]

        # Get low and high bounds of num_ctx
        low = split_info["num_ctx_min"]
        high = split_info["num_ctx_max"]

        # Check context points' distribution
        ctx_dist = split_info.get("ctx_dist", "uniform")

        # Match case
        if ctx_dist == "uniform":
            # Uniform distribution
            high = high + 1
            num_ctx = torch.randint(low=low,
                                    high=high,
                                    size=(1,),
                                    generator=rng)
        elif ctx_dist == "exponential":
            # Exponential distribution
            rate = -math.log(1e-1) / (high - low)
            exp = torch.distributions.exponential.Exponential(rate=rate,
                                                              validate_args=False)
            num_ctx = int(exp.sample().item() + low)
            if num_ctx > high:
                num_ctx = high
        else:
            raise ValueError("Unknown context case")

        # Debug only
        # num_ctx = 5

        assert num_ctx < num_all, "Number of context set must be a subset."

        # Determine context points indices
        if shuffle:
            index_pts = torch.randperm(n=num_all, generator=None)
        else:
            index_pts = list(range(num_all))

        return index_pts[:num_ctx], index_pts[num_ctx:]

    def _get_encoder_input(self):
        dict_encoder_input = dict()
        original_encoder_input = dict()

        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            # Split
            data_dict = self._data_dict_batch[data_key]
            if self._data_info[data_key]["time_dependent"]:
                ctx_dict = self._split_ctx(data_dict, is_traj=
                self._predict_key == data_key)
            else:
                ctx_dict = data_dict

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)

            # Dict to Tensor
            if self._data_info[data_key]["time_dependent"]:
                ctx_batch = torch.cat(
                    [norm_ctx_dict["time"], norm_ctx_dict["value"]],
                    dim=-1)
            else:
                ctx_batch = norm_ctx_dict["value"]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        """
        Get the normalized and original decoder input
        Returns:
            normalized and original decoder input
        """
        if self._inference_only:
            return None
        else:
            norm_decoder_input = \
                self._normalize_data(self._predict_key,
                                     {"time": self._decoder_input})["time"]
            return norm_decoder_input, self._decoder_input

    def _get_decoder_output_ground_truth(self):
        if self._inference_only:
            return None
        else:
            return self._decoder_output_ground_truth

    def _get_reconstructor_input(self):
        return None

    def _get_final_ground_truth(self):
        return self._get_decoder_output_ground_truth()


class CtxPredDiffAssignment(CtxPredSplitAssignment):
    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert prediction of raw trajectory
        assert self._traj_key == self._predict_key

    def _split_ctx(self, data_dict, is_traj=False):
        """
        Split context data from all data. If inference only, all data will be
        context data
        Args:
            data_dict: data to be split
            is_traj: if current data is also the final ground truth
        Returns:
            context data
        """
        time_batch = data_dict["time"]
        value_batch = data_dict["value"]
        ctx = dict()
        if not self._inference_only:
            ctx["time"] = time_batch[:, self._ctx_idx]
            ctx["value"] = value_batch[:, self._ctx_idx]

        else:
            ctx["time"] = time_batch
            ctx["value"] = value_batch
        return ctx

    def _get_decoder_input(self):
        """
        Get the normalized and original decoder input
        Returns:
            normalized and original decoder input
        """
        if self._inference_only:
            return None
        else:
            time_batch = self._data_dict_batch[self._traj_key]["time"]
            value_batch = self._data_dict_batch[self._traj_key]["value"]
            self._decoder_input = time_batch[:, self._pred_idx]
            self._decoder_output_ground_truth = value_batch[:,
                                                self._pred_idx]
            norm_decoder_input = \
                self._normalize_data(self._predict_key,
                                     {"time": self._decoder_input})["time"]
            return norm_decoder_input, self._decoder_input


class ProMPAssignment(DataAssignmentInterface):
    def __init__(self, **kwargs):
        super(ProMPAssignment, self).__init__(**kwargs)
        self._check_data_info()

        # Arguments for splitting dataset
        self._split_info = kwargs["split_info"]

        self._ctx_idx = None

    def assign_data(self, inference_only=False, **kwargs):
        if not inference_only:
            self._ctx_idx, _ = \
                CtxPredSplitAssignment.ctx_pred_index(self._split_info,
                                                      self._num_all,
                                                      self._rng)
        return super(ProMPAssignment, self).assign_data(inference_only,
                                                        **kwargs)

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert not to predict raw trajectory
        assert self._traj_key != self._predict_key

    def _split_ctx(self, data_dict):
        """
        Split context data from all data.
        Args:
            data_dict: data to be split

        Returns:
            context data
        """
        time_batch = data_dict["time"]
        value_batch = data_dict["value"]
        ctx = dict()
        if not self._inference_only:
            ctx["time"] = time_batch[:, self._ctx_idx]
            ctx["value"] = value_batch[:, self._ctx_idx]
        else:
            ctx["time"] = time_batch
            ctx["value"] = value_batch
        return ctx

    def _get_encoder_input(self):
        # Initialize a dictionary to store context data
        dict_encoder_input = dict()
        original_encoder_input = dict()

        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            data_dict = self._data_dict_batch[data_key]
            if self._data_info[data_key]["time_dependent"]:
                ctx_dict = self._split_ctx(data_dict)
            else:
                ctx_dict = data_dict

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)

            # Dict to Tensor
            if self._data_info[data_key]["time_dependent"]:
                ctx_batch = torch.cat([norm_ctx_dict["time"],
                                       norm_ctx_dict["value"]], dim=-1)
            else:
                ctx_batch = norm_ctx_dict["value"]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        return None, None

    def _get_decoder_output_ground_truth(self):
        if self._inference_only:
            return None
        else:
            return self._data_dict_batch[self._predict_key]["value"]

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None
        else:
            times = self._data_dict_batch[self._traj_key]["time"].squeeze(-1)
            return {"times": times}

    def _get_final_ground_truth(self):
        if self._inference_only:
            return None
        else:
            return self._data_dict_batch[self._traj_key]["value"]


class IDMPAssignment(DataAssignmentInterface):
    """
    context and trajectory are same data type
    """

    def __init__(self, **kwargs):
        super(IDMPAssignment, self).__init__(**kwargs)
        self._check_data_info()

        # Arguments for splitting dataset
        self._split_info = kwargs["split_info"]
        self.num_dof = kwargs["reconstructor_info"]["num_dof"]
        self._ctx_idx, self._pred_idx = None, None

    def assign_data(self, inference_only=False, **kwargs):
        if not inference_only:
            self._ctx_idx, self._pred_idx = \
                self.ctx_pred_index(self._split_info,
                                    self._num_all,
                                    self._rng,
                                    **kwargs)
        return super(IDMPAssignment, self).assign_data(inference_only, **kwargs)

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert not to predict raw trajectory
        assert self._traj_key != self._predict_key

    @staticmethod
    def ctx_pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute context and predict index for splitting
        Returns:
            None
        """
        assert split_info is not None, "Need context split info"

        # Get low and high bounds of num_ctx
        num_ctx_pred_pts = split_info["num_ctx_pred_pts"]

        # Some keyword arguments
        num_ctx = kwargs.get("num_ctx", None)
        if num_ctx is not None:
            assert num_ctx < num_ctx_pred_pts

        assert num_ctx_pred_pts < num_all, \
            "Number of context + predict set must be a subset."
        # Check context points' distribution

        # Select context + prediction set
        index_pts = torch.randperm(n=num_all, generator=None)
        index_ctx_pred = torch.sort(index_pts[:num_ctx_pred_pts])[0]

        # Generate ctx and pred indices
        poisson = \
            torch.distributions.poisson.Poisson(rate=num_ctx_pred_pts / 2,
                                                validate_args=False)

        if num_ctx is None:
            num_ctx = int(
                min(1 + poisson.sample().item(), num_ctx_pred_pts - 1))

        return index_ctx_pred[:num_ctx], index_ctx_pred[num_ctx:]

    def _split_ctx(self, data_dict, is_traj):
        """
        Split context data from all data, optionally split and save predict data
        too. If inference only, all data will be context data
        Args:
            data_dict: data to be split
            is_traj: if current data is also the final ground truth
        Returns:
            context data
        """
        time_batch = data_dict["time"]
        value_batch = data_dict["value"]
        ctx = dict()
        if not self._inference_only:
            ctx["time"] = time_batch[:, self._ctx_idx]
            ctx["value"] = value_batch[:, self._ctx_idx]
            if is_traj:
                self._mp_rec_input = dict()
                # Last context time and value are used as boundary condition
                self._mp_rec_input["bc_index"] = self._ctx_idx[-1][
                    None].expand(time_batch.shape[0])
                self._mp_rec_input["bc_pos"] = ctx["value"][:, -1,
                                               :self.num_dof]
                self._mp_rec_input["bc_vel"] = ctx["value"][:, -1,
                                               self.num_dof:]

                # The trajectory will be reconstructed at the time referred by
                # these index
                self._mp_rec_input["time_indices"] = \
                    self._pred_idx[None].expand(time_batch.shape[0], -1)

                # The ground-truth after reconstruction
                self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
                                            :self.num_dof]
        else:
            ctx["time"] = time_batch
            ctx["value"] = value_batch
        return ctx

    def _get_encoder_input(self):
        dict_encoder_input = dict()
        original_encoder_input = dict()

        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            data_dict = self._data_dict_batch[data_key]
            if self._data_info[data_key]["time_dependent"]:
                ctx_dict = self._split_ctx(data_dict, is_traj=
                self._traj_key == data_key)
            else:
                ctx_dict = data_dict

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)

            # Dict to Tensor
            if self._data_info[data_key]["time_dependent"]:
                ctx_batch = torch.cat([norm_ctx_dict["time"],
                                       norm_ctx_dict["value"]],
                                      dim=-1)
            else:
                ctx_batch = norm_ctx_dict["value"]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        return None, None

    def _get_decoder_output_ground_truth(self):
        if self._inference_only:
            return None
        else:
            return self._data_dict_batch[self._predict_key]["value"]

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None
        else:
            return self._mp_rec_input

    def _get_final_ground_truth(self):
        if self._inference_only:
            return None
        else:
            return self._mp_rec_ground_truth


class IDMPCtxTrajDiffAssignment(IDMPAssignment):
    """
    context and trajectory are not same data type
    """

    def _split_ctx(self, data_dict, is_traj):
        """
        Split context data from all data. If inference only, all data will be
        context data
        Args:
            data_dict: data to be split
            is_traj: if current data is also the final ground truth
        Returns:
            context data
        """
        time_batch = data_dict["time"]
        value_batch = data_dict["value"]
        ctx = dict()
        if not self._inference_only:
            ctx["time"] = time_batch[:, self._ctx_idx]
            ctx["value"] = value_batch[:, self._ctx_idx]

        else:
            ctx["time"] = time_batch
            ctx["value"] = value_batch
        return ctx

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None

        else:
            time_batch = self._data_dict_batch[self._traj_key]["time"]
            value_batch = self._data_dict_batch[self._traj_key]["value"]

            self._mp_rec_input = dict()
            # time and value at last ctx time step are used as boundary
            # condition
            # TODO, SHOULD WE REALLY NEED TO HAVE AT LEAST ONE CTX PT?
            self._mp_rec_input["bc_index"] = self._ctx_idx[-1][
                None].expand(time_batch.shape[0])
            self._mp_rec_input["bc_pos"] = value_batch[:, self._ctx_idx[-1],
                                           :self.num_dof]
            self._mp_rec_input["bc_vel"] = value_batch[:, self._ctx_idx[-1],
                                           self.num_dof:]

            # The trajectory will be reconstructed at the time referred by
            # these index
            self._mp_rec_input["time_indices"] = \
                self._pred_idx[None].expand(time_batch.shape[0], -1)

            # The ground-truth after reconstruction
            self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
                                        :self.num_dof]

            return self._mp_rec_input

    def _get_final_ground_truth(self):
        if self._inference_only:
            return None
        else:
            return self._mp_rec_ground_truth


class IDMPFixIntervalCtxTrajDiffAssignment(IDMPCtxTrajDiffAssignment):
    """
    Context and trajectory are not same data type
    Context comes in a fixed interval
    """

    @staticmethod
    def ctx_pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute context and predict index for splitting
        Returns:
            None
        """
        assert split_info is not None, "Need context split info"

        # Get low and high bounds of num_ctx
        num_ctx_pred_pts = split_info["num_ctx_pred_pts"]

        # Some keyword arguments
        num_ctx = kwargs.get("num_ctx", None)
        first_index = kwargs.get("first_index", None)
        if num_ctx is not None:
            assert num_ctx < num_ctx_pred_pts

        assert num_ctx_pred_pts < num_all, \
            "Number of context + predict set must be a subset."
        # Check context points' distribution

        interval = num_all // num_ctx_pred_pts
        num_residual = num_all % num_ctx_pred_pts

        if first_index is None:
            first_index = torch.randint(low=0, high=interval + num_residual,
                                        size=[], generator=rng).item()
        else:
            assert 0 <= first_index < interval + num_residual

        # Select context + prediction set
        index_ctx_pred = torch.arange(start=first_index, end=num_all,
                                      step=interval, dtype=torch.long)

        # Generate ctx and pred indices

        if num_ctx is None:
            uni_dist = \
                torch.distributions.uniform.Uniform(low=0,
                                                    high=len(index_ctx_pred),
                                                    validate_args=False)
            num_ctx = int(min(1 + uni_dist.sample().item(),
                              num_ctx_pred_pts - 1))

        return index_ctx_pred[:num_ctx], index_ctx_pred[num_ctx:]


class IDMPFixIntervalCtxRandPredTrajDiffAssignment(IDMPCtxTrajDiffAssignment):
    """
    Context and trajectory are not same data type
    Context comes in a fixed interval, Predictions pts are randomly selected
    """

    @staticmethod
    def ctx_pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute context and predict index for splitting
        Returns:
            None
        """
        assert split_info is not None, "Need context split info"

        # Get low and high bounds of num_ctx
        num_ctx_pred_pts = split_info["num_ctx_pred_pts"]

        # Some keyword arguments
        num_ctx = kwargs.get("num_ctx", None)
        first_index = kwargs.get("first_index", None)
        if num_ctx is not None:
            assert num_ctx < num_ctx_pred_pts

        assert num_ctx_pred_pts < num_all, \
            "Number of context + predict set must be a subset."
        # Check context points' distribution

        interval = num_all // num_ctx_pred_pts
        num_residual = num_all % num_ctx_pred_pts

        if first_index is None:
            first_index = torch.randint(low=0, high=interval + num_residual,
                                        size=[], generator=rng).item()
        else:
            assert 0 <= first_index < interval + num_residual

        # Select possible context set
        possible_index_ctx = torch.arange(start=first_index, end=num_all,
                                          step=interval, dtype=torch.long)

        # Generate ctx indices
        if num_ctx is None:
            uni_dist = \
                torch.distributions.uniform.Uniform(low=0,
                                                    high=len(
                                                        possible_index_ctx),
                                                    validate_args=False)
            # todo, -2 guarantees at least 2 pred pts
            num_ctx = int(min(1 + uni_dist.sample().item(),
                              num_ctx_pred_pts - 2))

        ctx_indices = possible_index_ctx[:num_ctx]

        # Generate pred indices
        num_pred = num_ctx_pred_pts - num_ctx
        index_perm_remain = torch.randperm(n=num_all - ctx_indices[-1] - 1,
                                           generator=None) + ctx_indices[-1] + 1
        pred_indices = index_perm_remain[:num_pred]
        return ctx_indices, pred_indices


class IDMPPairPredAssignment(IDMPFixIntervalCtxRandPredTrajDiffAssignment):
    @staticmethod
    def ctx_pred_index(split_info, num_all, rng, **kwargs):
        ctx_indices, pred_indices = \
            IDMPFixIntervalCtxRandPredTrajDiffAssignment.ctx_pred_index(
                split_info, num_all, rng, **kwargs)
        pred_pairs = torch.combinations(pred_indices, 2)
        return ctx_indices, pred_pairs

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None

        else:
            time_batch = self._data_dict_batch[self._traj_key]["time"]
            value_batch = self._data_dict_batch[self._traj_key]["value"]

            self._mp_rec_input = dict()
            # time and value at last ctx time step are used as boundary
            # condition
            bc_index = self._ctx_idx[-1]
            bc_pos = value_batch[:, self._ctx_idx[-1], :self.num_dof]
            bc_vel = value_batch[:, self._ctx_idx[-1], self.num_dof:]

            # Add axis to get shape
            # [num_traj, num_time_group, num_times=2, (optional) dim_data]
            self._mp_rec_input["bc_index"] = \
                util.add_expand_dim(data=bc_index, add_dim_indices=[0, 1],
                                    add_dim_sizes=[time_batch.shape[0],
                                                   self._pred_idx.shape[0]])
            self._mp_rec_input["bc_pos"] = \
                util.add_expand_dim(data=bc_pos, add_dim_indices=[1],
                                    add_dim_sizes=[self._pred_idx.shape[0]])
            self._mp_rec_input["bc_vel"] = \
                util.add_expand_dim(data=bc_vel, add_dim_indices=[1],
                                    add_dim_sizes=[self._pred_idx.shape[0]])

            # The trajectory will be reconstructed at the time referred by
            # these index groups!
            self._mp_rec_input["time_indices"] = \
                util.add_expand_dim(data=self._pred_idx, add_dim_indices=[0],
                                    add_dim_sizes=[time_batch.shape[0]])

            # The ground-truth after reconstruction
            self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
                                        :self.num_dof]

            return self._mp_rec_input


class DMPDiffAssignment(IDMPCtxTrajDiffAssignment):
    """
    context and trajectory are not same data type
    """

    @staticmethod
    def ctx_pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute context and predict index for splitting
        Returns:
            None
        """
        ctx_indices, _ = \
            IDMPFixIntervalCtxTrajDiffAssignment.ctx_pred_index(
                split_info, num_all, rng, **kwargs)
        pred_indices = torch.arange(start=ctx_indices[-1], end=num_all,
                                    step=1, dtype=torch.long)
        return ctx_indices, pred_indices

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None

        else:
            time_batch = self._data_dict_batch[self._traj_key]["time"]
            value_batch = self._data_dict_batch[self._traj_key]["value"]

            self._mp_rec_input = dict()
            # time and value at last ctx time step are used as boundary
            # condition
            self._mp_rec_input["bc_time"] = time_batch[:, self._ctx_idx[-1], 0]
            self._mp_rec_input["bc_pos"] = value_batch[:, self._ctx_idx[-1],
                                           :self.num_dof]
            self._mp_rec_input["bc_vel"] = value_batch[:, self._ctx_idx[-1],
                                           self.num_dof:]

            # The trajectory will be reconstructed at the time referred by
            # these index
            self._mp_rec_input["times"] = time_batch[:, self._pred_idx].squeeze(
                -1)

            # The ground-truth after reconstruction
            self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
                                        :self.num_dof]

            return self._mp_rec_input


class DMPImageAssignment(DataAssignmentInterface):
    """
    Assignment for image usage
    """

    def __init__(self, **kwargs):
        super(DMPImageAssignment, self).__init__(**kwargs)
        self._check_data_info()
        self._pred_idx = None
        self.num_dof = kwargs["reconstructor_info"]["num_dof"]
        self._split_info = kwargs["split_info"]

    def assign_data(self, inference_only=False, **kwargs):
        if not inference_only:
            self._pred_idx = \
                self.pred_index(self._split_info,
                                self._num_all,
                                self._rng,
                                **kwargs)
        return super(DMPImageAssignment, self).assign_data(inference_only,
                                                           **kwargs)

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert not to predict raw trajectory
        assert self._traj_key != self._predict_key

    @staticmethod
    def pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute predict index
        Returns:
            None
        """
        assert split_info is not None, "Need split info"

        # Get low and high bounds of num_ctx
        num_pred_pts = split_info["num_pred_pts"]

        assert num_pred_pts < num_all, \
            "Number of predict set must be a subset."

        # Select context + prediction set
        index_pts = torch.randperm(n=num_all, generator=None)
        index_pred = torch.sort(index_pts[:num_pred_pts])[0]

        return index_pred

    def _get_encoder_input(self):
        """
        Get encoder input (context), normalized
        Returns:
            dict_encoder_input: dictionary of encoder input
        """
        dict_encoder_input = dict()
        original_encoder_input = dict()
        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            ctx_dict = self._data_dict_batch[data_key]

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)
            ctx_batch = norm_ctx_dict["value"]

            # Remove original img in noise case, only use the first n
            # noisy images
            if ctx_batch.shape[1] > 1:
                ctx_batch = ctx_batch[:, :-1]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        """
        Get decoder input, normalized
        Returns:
            Input of the decoder, such as query time points
        """
        return None, None

    def _get_decoder_output_ground_truth(self):
        """
        Ground truth of the decoder prediction
        Returns:
            Ground truth of decoder prediction
        """
        if self._inference_only:
            return None
        else:
            return self._data_dict_batch[self._predict_key]["value"]

    def _get_reconstructor_input(self):
        """
        Input of trajectory reconstruction, such as time points
        Returns:
            Input for reconstruction
        """
        if self._inference_only:
            return None
        else:
            data_dict = self._data_dict_batch[self._traj_key]
            time_batch = data_dict["time"]
            value_batch = data_dict["value"]

            self._mp_rec_input = dict()
            self._mp_rec_input["bc_index"] = \
                torch.Tensor([0]).long().expand(time_batch.shape[0])

            self._mp_rec_input["bc_vel"] = torch.zeros([value_batch.shape[0],
                                                        self.num_dof])

            self._mp_rec_input["time_indices"] = \
                self._pred_idx[None,].expand(time_batch.shape[0], -1)

            # The ground-truth after reconstruction
            self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
                                        :self.num_dof]

            return self._mp_rec_input

    def _get_final_ground_truth(self):
        """
        Get ground truth after post-processing the decoder prediction
        Returns:
            Final ground truth
        """
        if self._inference_only:
            return None
        else:
            return self._mp_rec_ground_truth


class DMPImagePairedAssignment(DMPImageAssignment):
    @staticmethod
    def pred_index(split_info, num_all, rng, **kwargs):
        index_pred = \
            DMPImageAssignment.pred_index(split_info, num_all, rng, **kwargs)
        pred_pairs = torch.combinations(index_pred, 2)
        return pred_pairs

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None

        else:
            time_batch = self._data_dict_batch[self._traj_key]["time"]
            value_batch = self._data_dict_batch[self._traj_key]["value"]

            self._mp_rec_input = dict()

            bc_index = torch.Tensor([0]).long().squeeze()
            bc_vel = torch.zeros([self.num_dof])
            # time_indices = self._pred_idx[None,].expand(time_batch.shape[0], -1)

            # Add axis to get shape
            # [num_traj, num_time_group, num_times=2, (optional) dim_data]
            self._mp_rec_input["bc_index"] = \
                util.add_expand_dim(data=bc_index, add_dim_indices=[0, 1],
                                    add_dim_sizes=[time_batch.shape[0],
                                                   self._pred_idx.shape[0]])
            self._mp_rec_input["bc_vel"] = \
                util.add_expand_dim(data=bc_vel, add_dim_indices=[0, 1],
                                    add_dim_sizes=[value_batch.shape[0],
                                                   self._pred_idx.shape[0]])

            # The trajectory will be reconstructed at the time referred by
            # these index groups!
            self._mp_rec_input["time_indices"] = \
                util.add_expand_dim(data=self._pred_idx, add_dim_indices=[0],
                                    add_dim_sizes=[time_batch.shape[0]])

            # The ground-truth after reconstruction
            self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
                                        :self.num_dof]

            return self._mp_rec_input


class PAPAssignment(DataAssignmentInterface):
    """
    Assignment for toy task
    """

    def __init__(self, **kwargs):
        super(PAPAssignment, self).__init__(**kwargs)
        self._check_data_info()
        self._pred_idx = None
        self.num_dof = kwargs["reconstructor_info"]["num_dof"]
        self._split_info = kwargs["split_info"]

    def assign_data(self, inference_only=False, **kwargs):
        if not inference_only:
            ctx_dict = self._data_dict_batch["ctx"]["value"]
            kwargs["bc_index"] = ctx_dict[..., 0]
            self._pred_idx = \
                self.pred_index(self._split_info,
                                self._num_all,
                                self._rng,
                                **kwargs)
        return super(PAPAssignment, self).assign_data(inference_only,
                                                      **kwargs)

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert not to predict raw trajectory
        assert self._traj_key != self._predict_key

    @staticmethod
    def pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute predict index
        Returns:
            None
        """
        assert split_info is not None, "Need split info"

        # Get low and high bounds of num_ctx
        num_pred_pts = split_info["num_pred_pts"]
        bc_index = kwargs["bc_index"]

        assert torch.all(num_pred_pts < num_all - bc_index), \
            "Number of predict set must be a subset."

        shift = (torch.linspace(0, 1, num_pred_pts + 1) * (num_all - bc_index - 1))[:, 1:].long()
        min_shift = torch.min(shift).item()
        fix_term = torch.randint(0, min_shift + 1, size=[])
        index_pred = bc_index + shift - fix_term
        assert torch.all(bc_index <= index_pred)

        return index_pred

    def _get_encoder_input(self):
        """
        Get encoder input (context), normalized
        Returns:
            dict_encoder_input: dictionary of encoder input
        """
        dict_encoder_input = dict()
        original_encoder_input = dict()
        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            ctx_dict = self._data_dict_batch[data_key]

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)
            ctx_batch = norm_ctx_dict["value"]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        """
        Get decoder input, normalized
        Returns:
            Input of the decoder, such as query time points
        """
        return None, None

    def _get_decoder_output_ground_truth(self):
        """
        Ground truth of the decoder prediction
        Returns:
            Ground truth of decoder prediction
        """
        if self._inference_only:
            return None
        else:
            return self._data_dict_batch[self._predict_key]["value"]

    def _get_reconstructor_input(self):
        """
        Input of trajectory reconstruction, such as time points
        Returns:
            Input for reconstruction
        """
        if self._inference_only:
            return None
        else:
            data_dict = self._data_dict_batch[self._traj_key]
            time_batch = data_dict["time"]
            value_batch = data_dict["value"]

            ctx_dict = self._data_dict_batch["ctx"]["value"]
            bc_index = ctx_dict[0]
            bc_pos = ctx_dict[1:3]
            bc_vel = ctx_dict[3:5]

            self._mp_rec_input = dict()

            self._mp_rec_input["bc_index"] = \
                torch.Tensor([bc_index]).long().expand(time_batch.shape[0])

            self._mp_rec_input["bc_pos"] = \
                util.add_expand_dim(bc_pos, [0], [time_batch.shape[0]])

            self._mp_rec_input["bc_vel"] = \
                util.add_expand_dim(bc_vel, [0], [time_batch.shape[0]])

            self._mp_rec_input["time_indices"] = self._pred_idx

            # The ground-truth after reconstruction
            # self._mp_rec_ground_truth = value_batch[:, self._pred_idx,
            #                             :self.num_dof]
            raise NotImplementedError
            self._mp_rec_ground_truth = value_batch[self._pred_idx, :self.num_dof]

            return self._mp_rec_input

    def _get_final_ground_truth(self):
        """
        Get ground truth after post-processing the decoder prediction
        Returns:
            Final ground truth
        """
        if self._inference_only:
            return None
        else:
            return self._mp_rec_ground_truth


class PAPPairedAssignment(PAPAssignment):

    @staticmethod
    def pred_index(split_info, num_all, rng, **kwargs):
        index_pred = \
            PAPAssignment.pred_index(split_info, num_all, rng, **kwargs)

        pred_pairs = torch.zeros([index_pred.shape[0], index_pred.shape[1] * (index_pred.shape[1] - 1) // 2, 2]).long()
        for i, pred_idx in enumerate(index_pred):
            pred_pairs[i] = torch.combinations(pred_idx).long()

        return pred_pairs

    def _get_reconstructor_input(self):
        if self._inference_only:
            return None

        else:
            data_dict = self._data_dict_batch[self._traj_key]
            time_batch = data_dict["time"]
            value_batch = data_dict["value"]

            ctx_dict = self._data_dict_batch["ctx"]["value"]
            bc_index = ctx_dict[..., 0].long()
            bc_pos_vel = torch.gather(value_batch, 1, util.add_expand_dim(bc_index, [-1], [self.num_dof * 2]))
            bc_pos = bc_pos_vel[..., 0, :self.num_dof]
            bc_vel = bc_pos_vel[..., 0, self.num_dof:]

            self._mp_rec_input = dict()

            self._mp_rec_input["bc_index"] = bc_index.expand(-1, self._pred_idx.shape[1])

            self._mp_rec_input["bc_pos"] = \
                util.add_expand_dim(data=bc_pos, add_dim_indices=[1],
                                    add_dim_sizes=[self._pred_idx.shape[1]])

            self._mp_rec_input["bc_vel"] = \
                util.add_expand_dim(data=bc_vel, add_dim_indices=[1],
                                    add_dim_sizes=[self._pred_idx.shape[1]])

            self._mp_rec_input["time_indices"] = self._pred_idx

            # The ground-truth after reconstruction
            temp_idx = util.add_expand_dim(self._pred_idx.reshape(self._pred_idx.shape[0], -1), [-1],
                                           [self.num_dof * 2])
            self._mp_rec_ground_truth = \
                torch.gather(input=value_batch,
                             dim=1,
                             index=temp_idx).reshape(*self._pred_idx.shape, -1)[..., :self.num_dof]

            return self._mp_rec_input


class DMPImageReplaningAssignment(DataAssignmentInterface):
    """
    Assignment for image re-planing usage
    """

    def __init__(self, **kwargs):
        super(DMPImageReplaningAssignment, self).__init__(**kwargs)
        self._check_data_info()
        self.replan_bound = None
        self._pred_idx = None
        self.num_dof = kwargs["reconstructor_info"]["num_dof"]
        self._split_info = kwargs["split_info"]

    def assign_data(self, inference_only=False, **kwargs):
        if not inference_only:
            self.replan_bound, self._pred_idx = \
                self.pred_index(self._split_info,
                                self._num_all,
                                self._rng,
                                **kwargs)
        return super(DMPImageReplaningAssignment, self).assign_data(
            inference_only, **kwargs)

    @staticmethod
    def pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute predict index for all replan stages
        Returns:
            bounds and pred_idx
        """
        assert split_info is not None, "Need split info"

        # Get low and high bounds of num_ctx
        num_pred_pts = split_info["num_pred_pts"]
        replanstage = split_info["replanstage"]
        num_replan_stages = len(replanstage)
        stage_indices = torch.Tensor([0.0] + replanstage) * num_all
        stage_indices = stage_indices.long()
        replan_bound = list()
        pred_pairs_list = list()
        for i in range(num_replan_stages):
            idx_low = stage_indices[i]
            idx_replan_bc = stage_indices[i+1]-1
            idx_high = num_all

            assert num_pred_pts < idx_high - idx_low, \
                "Number of predict set must be a subset."
            replan_bound.append([idx_low, idx_replan_bc])

            # Select prediction set
            index_pts = torch.randperm(n=idx_high - idx_low, generator=None) \
                        + idx_low
            index_pred = torch.sort(index_pts[:num_pred_pts])[0]
            pred_pairs = torch.combinations(index_pred, 2)
            pred_pairs_list.append(pred_pairs)
        return replan_bound, pred_pairs_list

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert not to predict raw trajectory
        assert self._traj_key != self._predict_key

    def _get_encoder_input(self):
        """
        Get encoder input (context), normalized
        Returns:
            dict_encoder_input: dictionary of encoder input
        """
        dict_encoder_input = dict()
        original_encoder_input = dict()
        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            ctx_dict = self._data_dict_batch[data_key]

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)
            ctx_batch = norm_ctx_dict["value"]

            # Remove original img in noise case, only use the first n
            # noisy images
            if ctx_batch.shape[1] > 1:
                ctx_batch = ctx_batch[:, :-1]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        """
        Get decoder input, normalized
        Returns:
            Input of the decoder, such as query time points
        """
        return None, None

    def _get_decoder_output_ground_truth(self):
        """
        Ground truth of the decoder prediction
        Returns:
            Ground truth of decoder prediction
        """
        if self._inference_only:
            return None
        else:
            return self._data_dict_batch[self._predict_key]["value"]

    def _get_reconstructor_input(self):
        """
        Input of trajectory reconstruction, such as time points
        Returns:
            Input for reconstruction
        """
        if self._inference_only:
            return None
        else:
            time_batch = self._data_dict_batch[self._traj_key]["time"]
            value_batch = self._data_dict_batch[self._traj_key]["value"]

            self._mp_rec_input_list = list()
            self._mp_rec_ground_truth_list = list()
            for i, (pred_pairs, bound) in enumerate(zip(self._pred_idx,
                                                        self.replan_bound)):
                self._mp_rec_input = dict()
                bc_index = torch.Tensor([bound[0]]).long().squeeze()
                end_index = torch.Tensor([bound[1]]).long().squeeze()

                # bc index should be the last index of last execution step
                if bc_index != 0:
                    bc_index -= 1
                self._mp_rec_input["bc_index"] = bc_index
                self._mp_rec_input["end_index"] = end_index

                self._mp_rec_input["time_indices"] = \
                    util.add_expand_dim(data=pred_pairs,
                                        add_dim_indices=[0],
                                        add_dim_sizes=[time_batch.shape[0]])

                # The ground-truth after reconstruction
                self._mp_rec_ground_truth = value_batch[:, pred_pairs,
                                            :self.num_dof]

                self._mp_rec_input_list.append(self._mp_rec_input)
                self._mp_rec_ground_truth_list.append(self._mp_rec_ground_truth)

            return self._mp_rec_input_list

    def _get_final_ground_truth(self):
        """
        Get ground truth after post-processing the decoder prediction
        Returns:
            Final ground truth
        """
        if self._inference_only:
            return None
        else:
            return self._mp_rec_ground_truth_list


class CNMPImageAssignment(DataAssignmentInterface):
    def __init__(self, **kwargs):
        super(CNMPImageAssignment, self).__init__(**kwargs)
        self._check_data_info()
        # self._pred_idx = None
        # self.num_dof = kwargs["reconstructor_info"]["num_dof"]
        # self._split_info = kwargs["split_info"]

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert to predict raw trajectory!
        assert self._traj_key == self._predict_key

    def assign_data(self,
                    inference_only=False,
                    **kwargs):
        return super(CNMPImageAssignment,
                     self).assign_data(inference_only, **kwargs)

    def _get_encoder_input(self):
        """
        Get encoder input (context), normalized
        Returns:
            dict_encoder_input: dictionary of encoder input
        """
        dict_encoder_input = dict()
        original_encoder_input = dict()

        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            ctx_dict = self._data_dict_batch[data_key]

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)
            ctx_batch = norm_ctx_dict["value"]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True

        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        """
        Get decoder input, normalized
        Returns:
            Input of the decoder, such as query time points
        """
        if self._inference_only:
            return None
        else:
            data_dict = self._data_dict_batch[self._traj_key]
            self._decoder_input = data_dict["time"]
            self._decoder_output_ground_truth = data_dict["value"]

            norm_decoder_input = \
                self._normalize_data(self._predict_key,
                                     {"time": self._decoder_input})["time"]
            return norm_decoder_input, self._decoder_input

    def _get_decoder_output_ground_truth(self):
        """
        Ground truth of the decoder prediction
        Returns:
            Ground truth of decoder prediction
        """
        if self._inference_only:
            return None
        else:
            return self._decoder_output_ground_truth

    def _get_reconstructor_input(self):
        """
        Input of trajectory reconstruction, such as time points
        Returns:
            Input for reconstruction
        """
        return None

    def _get_final_ground_truth(self):
        """
        Get ground truth after post-processing the decoder prediction
        Returns:
            Final ground truth
        """
        return self._get_decoder_output_ground_truth()


class CNMP_PAP_Assignment(DataAssignmentInterface):

    def __init__(self, **kwargs):
        super(CNMP_PAP_Assignment, self).__init__(**kwargs)
        self._check_data_info()
        self._pred_idx = None
        self._split_info = kwargs["split_info"]

    def _check_data_info(self):
        # Assert there is no data playing both context and predict roles
        assert self._data_info[self._predict_key]["context"] is False

        # Assert to predict raw trajectory
        assert self._traj_key == self._predict_key

    def assign_data(self, inference_only=False, **kwargs):
        if not inference_only:
            ctx_dict = self._data_dict_batch["ctx"]["value"]
            kwargs["bc_index"] = ctx_dict[..., 0]
            self._pred_idx = \
                self.pred_index(self._split_info,
                                self._num_all,
                                self._rng,
                                **kwargs)
        return super(CNMP_PAP_Assignment, self).assign_data(inference_only,
                                                            **kwargs)

    @staticmethod
    def pred_index(split_info, num_all, rng, **kwargs):
        """
        Compute predict index
        Returns:
            None
        """
        assert split_info is not None, "Need split info"

        # Get low and high bounds of num_ctx
        num_pred_pts = split_info["num_pred_pts"]
        bc_index = kwargs["bc_index"]

        assert torch.all(num_pred_pts < num_all - bc_index), \
            "Number of predict set must be a subset."

        shift = (torch.linspace(0, 1, num_pred_pts + 1) * (num_all - bc_index - 1))[:, 1:].long()
        min_shift = torch.min(shift).item()
        fix_term = torch.randint(0, min_shift + 1, size=[])
        index_pred = bc_index + shift - fix_term
        assert torch.all(bc_index <= index_pred)

        return index_pred

    def _get_encoder_input(self):
        """
        Get encoder input (context), normalized
        Returns:
            dict_encoder_input: dictionary of encoder input
        """
        dict_encoder_input = dict()
        original_encoder_input = dict()
        for encoder_name, info in self._encoder_info.items():
            # Get encoder's desired input data key
            data_key = info["input"]

            # Assert data has not been assigned before
            assert self._assigned[data_key] is False
            assert self._data_info[data_key]["context"] is True

            ctx_dict = self._data_dict_batch[data_key]

            # Normalize
            norm_ctx_dict = self._normalize_data(key=data_key, data=ctx_dict)
            ctx_batch = norm_ctx_dict["value"]

            # Store context batch of each encoder
            dict_encoder_input[encoder_name] = ctx_batch
            original_encoder_input[encoder_name] = ctx_dict

            # Change assigned flag
            self._assigned[data_key] = True
        return dict_encoder_input, original_encoder_input

    def _get_decoder_input(self):
        """
        Get decoder input, normalized
        Returns:
            Input of the decoder, such as query time points
        """
        if self._inference_only:
            return None, None
        else:
            data_dict = self._data_dict_batch[self._traj_key]

            temp_idx = self._pred_idx[..., None].long()
            self._decoder_input = torch.gather(input=data_dict["time"],
                                               dim=1,
                                               index=temp_idx)

            temp_idx = temp_idx.expand(*temp_idx.shape[:-1], data_dict["value"].shape[-1])
            self._decoder_output_ground_truth = torch.gather(input=data_dict["value"],
                                                             dim=1,
                                                             index=temp_idx)

            norm_decoder_input = \
                self._normalize_data(self._predict_key,
                                     {"time": self._decoder_input})["time"]

            return norm_decoder_input, self._decoder_input

    def _get_decoder_output_ground_truth(self):
        """
        Ground truth of the decoder prediction
        Returns:
            Ground truth of decoder prediction
        """
        if self._inference_only:
            return None
        else:
            return self._decoder_output_ground_truth

    def _get_reconstructor_input(self):
        """
        Input of trajectory reconstruction, such as time points
        Returns:
            Input for reconstruction
        """
        return None

    def _get_final_ground_truth(self):
        """
        Get ground truth after post-processing the decoder prediction
        Returns:
            Final ground truth
        """
        return self._get_decoder_output_ground_truth()


def get_assignment_strategy_dict():
    DATA_ASSIGN_DICT = {"CtxPredSplitAssignment": CtxPredSplitAssignment,
                        "CtxPredDiffAssignment": CtxPredDiffAssignment,
                        "ProMPAssignment": ProMPAssignment,
                        "IDMPAssignment": IDMPAssignment,
                        "IDMPCtxTrajDiffAssignment": IDMPCtxTrajDiffAssignment,
                        "IDMPFixIntervalCtxTrajDiffAssignment":
                            IDMPFixIntervalCtxTrajDiffAssignment,
                        "IDMPFixIntervalCtxRandPredTrajDiffAssignment":
                            IDMPFixIntervalCtxRandPredTrajDiffAssignment,
                        "IDMPPairPredAssignment": IDMPPairPredAssignment,
                        "DMPDiffAssignment": DMPDiffAssignment,
                        "DMPImageAssignment": DMPImageAssignment,
                        "DMPImagePairedAssignment": DMPImagePairedAssignment,
                        "PAPAssignment": PAPAssignment,
                        "PAPPairedAssignment": PAPPairedAssignment,
                        "CNMPImageAssignment": CNMPImageAssignment,
                        "DMPImageReplaningAssignment":
                            DMPImageReplaningAssignment,
                        "CNMP_PAP_Assignment": CNMP_PAP_Assignment}

    return DATA_ASSIGN_DICT
