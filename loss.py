"""
--
@brief:     Custom loss functions in PyTorch
"""
from torch import nn

from nmp.data_process import *
from nmp.mp import *


def log_likelihood(true_val,
                   pred_mean,
                   pred_diag=None,
                   pred_off_diag=None,
                   pred_L=None,
                   pred_cov=None):
    """
    Log likelihood, calculates log probability of the true target value of the
    target point according to the predicted multivariate normal distribution
    Args:
        true_val: true target values
        pred_mean: predicted mean of the Normal distribution
        pred_diag: predicted diagonal elements in Cholesky Decomposition
        pred_off_diag: predicted off-diagonal elements in Cholesky Decomposition
        pred_L: alternatively, use predicted Cholesky Decomposition
        pred_cov: alternatively, use predicted Covariance Matrix

    Returns:
        log likelihood
    """
    # Shape of true_val:
    # [num_traj, num_agg, num_time_pts, dim_val]
    #
    # Shape of pred_mean:
    # [num_traj, num_agg, num_time_pts, dim_val]
    #
    # Shape of pred_diag:
    # [num_traj, num_agg, num_time_pts, dim_val]
    #
    # Shape of pred_off_diag:
    # [num_traj, num_agg, num_time_pts, (dim_val * (dim_val - 1) // 2)]
    #
    # Shape of pred_L:
    # [num_traj, num_agg, num_time_pts, dim_val, dim_val]
    #
    # Shape of pred_cov:
    # [num_traj, num_agg, num_time_pts, dim_val, dim_val]
    #
    # Shape of output log likelihood:
    # [], a scalar

    # Case select, cov or Cholesky
    if pred_cov is not None:
        assert pred_diag is None and pred_off_diag is None and pred_L is None

        # Construct distribution
        mvn = MultivariateNormal(loc=pred_mean, covariance_matrix=pred_cov,
                                 validate_args=False)

    else:
        assert (pred_diag is None and pred_off_diag is None) != (pred_L is None)
        # Covariance Cholesky diagonal elements should be all positive
        if pred_diag is not None:
            assert pred_diag.min() > 0
            # Construct the Cholesky Decomposition Matrix, i.e. L
            L = util.build_lower_matrix(pred_diag, pred_off_diag)
        else:
            L_diag = torch.einsum('...ii->...i', pred_L)
            assert L_diag.min() > 0
            L = pred_L

        # Construct distribution, fast
        mvn = MultivariateNormal(loc=pred_mean, scale_tril=L,
                                 validate_args=False)

    # Compute log likelihood loss for each trajectory
    ll = mvn.log_prob(true_val).mean()

    # Return
    return ll


def marginal_log_likelihood(true_val,
                            pred_mean,
                            pred_diag=None,
                            pred_off_diag=None,
                            pred_L=None,
                            pred_cov=None):
    """
    Log likelihood, calculates log probability of the true target value of the
    target point according to the predicted multivariate normal distribution
    Args:
        true_val: true target values
        pred_mean: predicted mean of the Normal distribution
        pred_diag: predicted diagonal elements in Cholesky Decomposition
        pred_off_diag: predicted off-diagonal elements in Cholesky Decomposition
        pred_L: alternatively, use predicted Cholesky Decomposition
        pred_cov: alternatively, use predicted Covariance Matrix

    Returns:
        log likelihood
    """
    # Shape of true_val:
    # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
    #
    # Shape of pred_mean:
    # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
    #
    # Shape of pred_diag:
    # [num_traj, num_agg, num_smp, num_time_pts, dim_val]
    #
    # Shape of pred_off_diag:
    # [num_traj, num_agg, num_smp, num_time_pts, (dim_val * (dim_val - 1) // 2)]
    #
    # Shape of pred_L:
    # [num_traj, num_smp, num_time_pts, dim_val, dim_val]
    #
    # Shape of pred_cov:
    # [num_traj, num_smp, num_time_pts, dim_val, dim_val]
    #
    # Shape of output log likelihood:
    # [], a scalar

    # Extract numbers, dimensions
    num_traj = pred_mean.shape[0]
    num_agg = pred_mean.shape[1]
    num_smp = pred_mean.shape[2]
    num_time_pts = pred_mean.shape[3]

    # Case select, cov or Cholesky
    if pred_cov is not None:
        assert pred_diag is None and pred_off_diag is None and pred_L is None
        # Construct distribution
        mvn = MultivariateNormal(loc=pred_mean, covariance_matrix=pred_cov,
                                 validate_args=False)

    else:
        # XOR
        assert (pred_diag is None and pred_off_diag is None) != (pred_L is None)
        # Covariance Cholesky diagonal elements should be all positive
        if pred_diag is not None:
            assert pred_diag.min() > 0
            # Construct the Cholesky Decomposition Matrix, i.e. L
            L = util.build_lower_matrix(pred_diag, pred_off_diag)
        else:
            L_diag = torch.einsum('...ii->...i', pred_L)
            assert L_diag.min() > 0
            L = pred_L

        # Construct distribution, fast
        mvn = MultivariateNormal(loc=pred_mean, scale_tril=L,
                                 validate_args=False)

    # Compute log likelihood loss for each trajectory
    ll = mvn.log_prob(true_val)

    # Sum over time points
    ll = torch.sum(ll, dim=3)

    # MC average
    ll = torch.logsumexp(ll, dim=2)

    # Sum over trajectories
    ll = torch.sum(ll, dim=(0, 1))
    assert ll.ndim == 0

    # Get marginal log-likelihood loss
    ll = -num_traj * num_agg * np.log(num_smp) + ll
    mll = ll / (num_traj * num_agg * num_time_pts)

    # Return
    return mll


def nll_loss_(true_val,
              pred_mean,
              pred_diag=None,
              pred_off_diag=None,
              pred_L=None,
              pred_cov=None,
              **kwargs):
    return -log_likelihood(true_val,
                           pred_mean,
                           pred_diag=pred_diag,
                           pred_off_diag=pred_off_diag,
                           pred_L=pred_L,
                           pred_cov=pred_cov)


def nll_loss(true_val,
             pred_mean,
             pred_diag=None,
             pred_off_diag=None,
             pred_L=None,
             pred_cov=None,
             **kwargs):
    # External call wrapper of negative log-likelihood loss, will add
    # aggregation dimension to ground-truth

    num_agg = pred_mean.shape[1]
    true_val = true_val[:, None, :, :].expand(-1, num_agg, -1, -1)

    return nll_loss_(true_val,
                     pred_mean,
                     pred_diag=pred_diag,
                     pred_off_diag=pred_off_diag,
                     pred_L=pred_L,
                     pred_cov=pred_cov,
                     **kwargs)


def nmll_loss_(true_val,
               pred_mean,
               pred_diag=None,
               pred_off_diag=None,
               pred_L=None,
               pred_cov=None,
               **kwargs):
    # Loss
    return -marginal_log_likelihood(true_val,
                                    pred_mean,
                                    pred_diag=pred_diag,
                                    pred_off_diag=pred_off_diag,
                                    pred_L=pred_L,
                                    pred_cov=pred_cov)


def nmll_loss(true_val,
              pred_mean,
              pred_diag=None,
              pred_off_diag=None,
              pred_L=None,
              pred_cov=None,
              **kwargs):
    # External call wrapper of negative marginal-log-likelihood loss, will add
    # aggregation and mc-sample dimensions to ground-truth

    num_agg = pred_mean.shape[1]
    num_smp = pred_mean.shape[2]
    true_val = true_val[:, None, None, :, :].expand(-1, num_agg, num_smp, -1,
                                                    -1)

    # Loss
    return nmll_loss_(true_val,
                      pred_mean,
                      pred_diag=pred_diag,
                      pred_off_diag=pred_off_diag,
                      pred_L=pred_L,
                      pred_cov=pred_cov,
                      **kwargs)


def mse_loss_(true_val, pred):
    mse = nn.MSELoss()
    return mse(pred, true_val)


def mp_rec_mse_loss(true_val,
                    pred_mean,
                    pred_diag=None,
                    pred_off_diag=None,
                    pred_L=None,
                    pred_cov=None,
                    **kwargs):
    """
    Mean square error loss
    Args:
        true_val: true target values
        pred_mean: predicted mean of the Normal distribution
        pred_diag: place holder for diagonal elements in Cholesky
        pred_off_diag: place holder for off-diagonal elements in Cholesky
        pred_L: place holder for Cholesky
        pred_cov: predicted Covariance Matrix
        **kwargs: keyword arguments for loss computation
    Returns:
        Mean square error loss

    """
    # Get keyword arguments
    normalizer = kwargs["normalizer"]
    mp_reconstructor = kwargs["reconstructor"]
    reconstructor_input = kwargs["reconstructor_input"]
    final_ground_truth = kwargs["final_ground_truth"]

    assert pred_mean.ndim == 4
    num_agg = pred_mean.shape[1]

    # Get global time and duration
    time_min = normalizer["time"]["min"].item()
    time_max = normalizer["time"]["max"].item()
    duration = time_max - time_min

    # Get reconstruction time and target_batch_values
    if reconstructor_input is not None and final_ground_truth is not None:
        # Expand dimension to reconstructor_input and final_ground_truth
        for key, value in reconstructor_input.items():
            v = util.add_expand_dim(data=value,
                                    add_dim_indices=[1],
                                    add_dim_sizes=[num_agg])
            reconstructor_input[key] = v

        final_ground_truth = util.add_expand_dim(data=final_ground_truth,
                                                 add_dim_indices=[1],
                                                 add_dim_sizes=[num_agg])

    elif reconstructor_input is None and final_ground_truth is None:
        # Raw time and trajectory are not available
        # Use synthetic reconstruction time
        raise NotImplementedError

    else:
        raise RuntimeError("Wrong combination of ground-truth times and values")

    # Get the predicted data dimension prepared, remove the dimension of time
    # weights is time-invariant
    pred_mean = pred_mean.squeeze(-2)

    # Compute loss
    reconstructor_input["std"] = False
    # Reconstruct predicted trajectories
    traj_mean, traj_cov = \
        mp_reconstructor.reconstruct(duration=duration,
                                     w_mean=pred_mean,
                                     w_L=None,
                                     **reconstructor_input)
    assert traj_cov is None
    if final_ground_truth.ndim > traj_mean.ndim:
        final_ground_truth = torch.einsum('...ij->...ji', final_ground_truth)
        final_ground_truth = final_ground_truth.reshape(
            *final_ground_truth.shape[:-2], -1)
    loss = mse_loss_(true_val=final_ground_truth, pred=traj_mean)

    # Return
    return loss


def mp_rec_digit_mse_loss(true_val,
                          pred_mean,
                          pred_diag=None,
                          pred_off_diag=None,
                          pred_L=None,
                          pred_cov=None,
                          **kwargs):
    # Get keyword arguments
    normalizer = kwargs["normalizer"]
    mp_reconstructor = kwargs["reconstructor"]
    num_dof = mp_reconstructor.get_config()["num_dof"]
    reconstructor_input = kwargs["reconstructor_input"]
    final_ground_truth = kwargs["final_ground_truth"]

    # Extract initial pos
    bc_pos = pred_mean[..., 0, :num_dof]
    pred_mean = pred_mean[..., num_dof:]
    kwargs["reconstructor_input"]["bc_pos"] = bc_pos

    assert pred_mean.ndim == 4
    num_agg = pred_mean.shape[1]

    # Get global time and duration
    time_min = normalizer["time"]["min"].item()
    time_max = normalizer["time"]["max"].item()
    duration = time_max - time_min

    # Get reconstruction time and target_batch_values
    if reconstructor_input is not None and final_ground_truth is not None:
        # Expand dimension to reconstructor_input and final_ground_truth
        for key, value in reconstructor_input.items():
            if key == "bc_pos":
                continue
            v = util.add_expand_dim(data=value,
                                    add_dim_indices=[1],
                                    add_dim_sizes=[num_agg])
            reconstructor_input[key] = v

        final_ground_truth = util.add_expand_dim(data=final_ground_truth,
                                                 add_dim_indices=[1],
                                                 add_dim_sizes=[num_agg])

    elif reconstructor_input is None and final_ground_truth is None:
        # Raw time and trajectory are not available
        # Use synthetic reconstruction time
        raise NotImplementedError

    else:
        raise RuntimeError("Wrong combination of ground-truth times and values")

    # Get the predicted data dimension prepared, remove the dimension of time
    # weights is time-invariant
    pred_mean = pred_mean.squeeze(-2)

    # Compute loss
    reconstructor_input["std"] = False
    # Reconstruct predicted trajectories
    traj_mean, traj_cov = \
        mp_reconstructor.reconstruct(duration=duration,
                                     w_mean=pred_mean,
                                     w_L=None,
                                     **reconstructor_input)
    assert traj_cov is None
    if final_ground_truth.ndim > traj_mean.ndim:
        final_ground_truth = torch.einsum('...ij->...ji', final_ground_truth)
        final_ground_truth = final_ground_truth.reshape(
            *final_ground_truth.shape[:-2], -1)
    loss = mse_loss_(true_val=final_ground_truth, pred=traj_mean)

    # Return
    return loss


def mp_rec_ll_loss(true_val,
                   pred_mean,
                   pred_diag=None,
                   pred_off_diag=None,
                   pred_L=None,
                   pred_cov=None,
                   **kwargs):
    """
    Reconstruct trajectories using mp weights and then compute likelihood loss
    Args:
        true_val: true mp weights
        pred_mean: predicted mean of MP weights
        pred_diag: predicted diagonal of MP weights
        pred_off_diag: predicted off-diagonal of MP weights
        pred_L: predicted Cholesky of MP weights
        pred_cov: predicted Covariance Matrix
        **kwargs: keyword arguments for loss computation

    Returns:
        loss

    """
    # Get keyword arguments
    normalizer = kwargs["normalizer"]
    mp_reconstructor = kwargs["reconstructor"]
    reconstructor_input = kwargs["reconstructor_input"]
    final_ground_truth = kwargs["final_ground_truth"]

    assert (pred_diag is None) != (pred_L is None)

    # Check Monte Carlo
    if pred_mean.ndim == 4:
        num_traj, num_agg = pred_mean.shape[:2]
        num_smp = None
    elif pred_mean.ndim == 5:
        num_traj, num_agg, num_smp = pred_mean.shape[:3]
    else:
        raise RuntimeError("Unknown weights dimension")

    # Get global time and duration
    time_min = normalizer["time"]["min"].item()
    time_max = normalizer["time"]["max"].item()
    duration = time_max - time_min

    # Get reconstruction time and target_batch_values
    if reconstructor_input is not None and final_ground_truth is not None:
        # Expand dimension to reconstructor_input and final_ground_truth
        for key, value in reconstructor_input.items():
            # Loop over key
            if num_smp is not None:
                v = util.add_expand_dim(data=value,
                                        add_dim_indices=[1, 2],
                                        add_dim_sizes=[num_agg, num_smp])
            else:
                v = util.add_expand_dim(data=value,
                                        add_dim_indices=[1],
                                        add_dim_sizes=[num_agg])
            reconstructor_input[key] = v

        final_ground_truth = util.add_expand_dim(data=final_ground_truth,
                                                 add_dim_indices=[1],
                                                 add_dim_sizes=[num_agg])
        if num_smp is not None:
            final_ground_truth = util.add_expand_dim(data=final_ground_truth,
                                                     add_dim_indices=[2],
                                                     add_dim_sizes=[num_smp])

    elif reconstructor_input is None and final_ground_truth is None:
        # Raw time and trajectory are not available
        # Use synthetic reconstruction time
        raise NotImplementedError

    else:
        raise RuntimeError("Wrong combination of ground-truth times and values")

    # Get the predicted data dimension prepared, remove the dimension of time
    # weights is time-invariant
    pred_mean = pred_mean.squeeze(-2)
    pred_L = pred_L.squeeze(-3)

    # Add dim of time group if necessary
    if pred_mean.ndim < final_ground_truth.ndim - 1:
        num_time_group = final_ground_truth.shape[-3]
        final_ground_truth = torch.einsum('...ij->...ji', final_ground_truth)
        final_ground_truth = final_ground_truth.reshape(
            *final_ground_truth.shape[:-2], -1)

        pred_mean = util.add_expand_dim(data=pred_mean,
                                        add_dim_indices=[-2],
                                        add_dim_sizes=[num_time_group])
        pred_L = util.add_expand_dim(data=pred_L,
                                     add_dim_indices=[-3],
                                     add_dim_sizes=[num_time_group])

    # Compute loss
    if num_smp is None:
        # Vanilla case
        # Get full covariance of the predicted trajectory distribution
        reconstructor_input["std"] = False
        # Reconstruct predicted trajectories
        traj_mean, traj_cov = \
            mp_reconstructor.reconstruct(duration=duration,
                                         w_mean=pred_mean,
                                         w_L=pred_L,
                                         **reconstructor_input)

        loss = nll_loss_(true_val=final_ground_truth,
                         pred_mean=traj_mean,
                         pred_diag=None,
                         pred_off_diag=None,
                         pred_L=None,
                         pred_cov=traj_cov)

    else:
        # Monte-Carlo case
        # Get the standard deviation of the predicted trajectory distribution
        reconstructor_input["std"] = True

        # Reconstruct predicted trajectories
        traj_mean, traj_std = \
            mp_reconstructor.reconstruct(duration=duration,
                                         w_mean=pred_mean,
                                         w_L=pred_L,
                                         **reconstructor_input)

        loss = nmll_loss_(true_val=final_ground_truth,
                          pred_mean=traj_mean,
                          pred_diag=traj_std,
                          pred_off_diag=None,
                          pred_L=None,
                          pred_cov=None)

    # Return
    return loss


def mp_rec_digit_ll_loss(true_val,
                         pred_mean,
                         pred_diag=None,
                         pred_off_diag=None,
                         pred_L=None,
                         pred_cov=None,
                         **kwargs):
    # Get keyword arguments
    normalizer = kwargs["normalizer"]
    mp_reconstructor = kwargs["reconstructor"]
    num_dof = mp_reconstructor.get_config()["num_dof"]
    reconstructor_input = kwargs["reconstructor_input"]
    final_ground_truth = kwargs["final_ground_truth"]

    # If the init_pos and dmp are jointly distributed
    # joint_conditioning = True
    joint_conditioning = False
    if joint_conditioning:
        bc_pos_mean = pred_mean[..., :num_dof]
        bc_pos_L = pred_L[..., :num_dof, :num_dof]

        # Apply re-parameterization trick
        diag = torch.ones_like(bc_pos_mean)
        L = util.build_lower_matrix(diag, None)
        mvn = MultivariateNormal(loc=torch.zeros_like(bc_pos_mean),
                                 scale_tril=L,
                                 validate_args=False)
        epsilon = mvn.sample()
        sample_bc_pos = \
            torch.einsum('...ik,...k->...i', bc_pos_L, epsilon) + bc_pos_mean

        pred_mean, pred_L = \
            util.joint_to_conditional(pred_mean, pred_L, sample_bc_pos)
        bc_pos = sample_bc_pos[..., 0, :]

    else:
        # Extract initial pos, and remove init pos from prediction
        bc_pos = pred_mean[..., 0, :num_dof]
        pred_mean = pred_mean[..., num_dof:]
        pred_L = pred_L[..., num_dof:, num_dof:]

    assert pred_mean.ndim == 4
    num_agg = pred_mean.shape[1]

    # Get global time and duration
    time_min = normalizer["time"]["min"].item()
    time_max = normalizer["time"]["max"].item()
    duration = time_max - time_min

    # Get reconstruction time and target_batch_values
    if reconstructor_input is not None and final_ground_truth is not None:
        # Expand dimension to reconstructor_input and final_ground_truth
        for key, value in reconstructor_input.items():
            if key == "bc_pos":
                continue
            v = util.add_expand_dim(data=value,
                                    add_dim_indices=[1],
                                    add_dim_sizes=[num_agg])
            reconstructor_input[key] = v

        final_ground_truth = util.add_expand_dim(data=final_ground_truth,
                                                 add_dim_indices=[1],
                                                 add_dim_sizes=[num_agg])

    elif reconstructor_input is None and final_ground_truth is None:
        # Raw time and trajectory are not available
        # Use synthetic reconstruction time
        raise NotImplementedError

    else:
        raise RuntimeError("Wrong combination of ground-truth times and values")

    # Get the predicted data dimension prepared, remove the dimension of time
    # weights is time-invariant
    pred_mean = pred_mean.squeeze(-2)
    pred_L = pred_L.squeeze(-3)

    # Add dim of time group if necessary
    if pred_mean.ndim < final_ground_truth.ndim - 1:
        num_time_group = final_ground_truth.shape[-3]
        final_ground_truth = torch.einsum('...ij->...ji', final_ground_truth)
        final_ground_truth = final_ground_truth.reshape(
            *final_ground_truth.shape[:-2], -1)

        pred_mean = util.add_expand_dim(data=pred_mean,
                                        add_dim_indices=[-2],
                                        add_dim_sizes=[num_time_group])
        pred_L = util.add_expand_dim(data=pred_L,
                                     add_dim_indices=[-3],
                                     add_dim_sizes=[num_time_group])
        bc_pos = util.add_expand_dim(data=bc_pos,
                                     add_dim_indices=[-2],
                                     add_dim_sizes=[num_time_group])

    reconstructor_input["bc_pos"] = bc_pos

    # Compute loss
    reconstructor_input["std"] = False
    # Reconstruct predicted trajectories
    traj_mean, traj_cov = \
        mp_reconstructor.reconstruct(duration=duration,
                                     w_mean=pred_mean,
                                     w_L=pred_L,
                                     **reconstructor_input)

    loss = nll_loss_(true_val=final_ground_truth,
                     pred_mean=traj_mean,
                     pred_diag=None,
                     pred_off_diag=None,
                     pred_L=None,
                     pred_cov=traj_cov)

    return loss


def mp_rec_digit_replan_ll_loss(true_val,
                                pred_mean,
                                pred_diag=None,
                                pred_off_diag=None,
                                pred_L=None,
                                pred_cov=None,
                                **kwargs):
    # Shape of pred_mean
    # [num_traj, num_agg, num_time=1, dim_data=dim_x_y + dmp_params]
    #
    # Shape of pred_L
    # [num_traj, num_agg, num_time=1, dim_data, dim_data]

    # Get keyword arguments
    normalizer = kwargs["normalizer"]
    mp_reconstructor = kwargs["reconstructor"]
    num_dof = mp_reconstructor.get_config()["num_dof"]
    reconstructor_input = kwargs["reconstructor_input"]
    final_ground_truth = kwargs["final_ground_truth"]

    ############################################################################
    use_last_mean = False
    end_pos_mean, end_vel_mean = None, None
    ############################################################################

    num_replan_steps = len(reconstructor_input)
    total_loss = 0

    # Get global time and duration
    time_min = normalizer["time"]["min"].item()
    time_max = normalizer["time"]["max"].item()
    duration = time_max - time_min

    for step in range(num_replan_steps):
        # Get reconstruction prediction for current step
        curr_pred_mean = pred_mean[:, step + 1, ...]
        curr_pred_L = pred_L[:, step + 1, ...]

        # Get reconstruction input for current step
        curr_rec_input = reconstructor_input[step]
        curr_bc_index = curr_rec_input["bc_index"]
        curr_end_index = curr_rec_input["end_index"]

        # Dynamically add boundary pos and vel
        if step == 0:
            curr_bc_pos = curr_pred_mean[..., 0, :num_dof]
            curr_bc_vel = torch.zeros_like(curr_bc_pos)

        else:
            curr_bc_pos = end_pos_mean
            curr_bc_vel = end_vel_mean

        curr_pred_mean = curr_pred_mean[..., num_dof:]
        curr_pred_L = curr_pred_L[..., num_dof:, num_dof:]
        assert curr_pred_mean.ndim == 3

        # Get the predicted data dimension prepared, remove the dimension of time
        # weights is time-invariant
        curr_pred_mean = curr_pred_mean.squeeze(-2)
        curr_pred_L = curr_pred_L.squeeze(-3)

        # Get the last robot state in current replan-step
        end_rec_input = dict()
        curr_bc_index = \
            util.add_expand_dim(data=curr_bc_index, add_dim_indices=[0],
                                add_dim_sizes=[curr_pred_mean.shape[0]])
        curr_end_index = \
            util.add_expand_dim(data=curr_end_index, add_dim_indices=[0, -1],
                                add_dim_sizes=[curr_pred_mean.shape[0], 1])
        end_rec_input["bc_index"] = curr_bc_index
        end_rec_input["bc_pos"] = curr_bc_pos
        end_rec_input["bc_vel"] = curr_bc_vel
        end_rec_input["get_velocity"] = True
        end_rec_input["time_indices"] = curr_end_index

        if use_last_mean:
            # Use mean of the prediction
            end_pos_mean, _, end_vel_mean, __ = \
                mp_reconstructor.reconstruct(duration=duration,
                                             w_mean=curr_pred_mean,
                                             w_L=curr_pred_L,
                                             **end_rec_input)

        else:
            # Use one sample of the prediction
            L = util.build_lower_matrix(torch.ones_like(curr_pred_mean), None)
            mvn = MultivariateNormal(loc=torch.zeros_like(curr_pred_mean),
                                     scale_tril=L, validate_args=False)
            epsilon = mvn.sample()
            sample = torch.einsum('...ik,...k->...i',
                                  curr_pred_L, epsilon) + curr_pred_mean

            end_pos_mean, _, end_vel_mean, __ = \
                mp_reconstructor.reconstruct(duration=duration,
                                             w_mean=sample,
                                             w_L=curr_pred_L,
                                             **end_rec_input)

        # Get reconstruction ground truth for current step
        curr_final_ground_truth = final_ground_truth[step]

        # Add dim of time group for loss computation
        if curr_pred_mean.ndim < curr_final_ground_truth.ndim - 1:
            num_time_group = curr_final_ground_truth.shape[-3]
            curr_final_ground_truth = torch.einsum('...ij->...ji',
                                                   curr_final_ground_truth)
            curr_final_ground_truth = curr_final_ground_truth.reshape(
                *curr_final_ground_truth.shape[:-2], -1)

            curr_pred_mean = util.add_expand_dim(data=curr_pred_mean,
                                                 add_dim_indices=[-2],
                                                 add_dim_sizes=[num_time_group])
            curr_pred_L = util.add_expand_dim(data=curr_pred_L,
                                              add_dim_indices=[-3],
                                              add_dim_sizes=[num_time_group])

            curr_bc_index = \
                util.add_expand_dim(data=curr_bc_index, add_dim_indices=[1],
                                    add_dim_sizes=[num_time_group])

            curr_bc_pos = util.add_expand_dim(data=curr_bc_pos,
                                              add_dim_indices=[-2],
                                              add_dim_sizes=[num_time_group])

            curr_bc_vel = util.add_expand_dim(data=curr_bc_vel,
                                              add_dim_indices=[-2],
                                              add_dim_sizes=[num_time_group])
        else:
            raise NotImplementedError

        curr_rec_input["bc_index"] = curr_bc_index
        curr_rec_input["bc_pos"] = curr_bc_pos
        curr_rec_input["bc_vel"] = curr_bc_vel

        # Compute loss
        curr_rec_input["std"] = False

        # Reconstruct predicted trajectories
        curr_traj_mean, curr_traj_cov = \
            mp_reconstructor.reconstruct(duration=duration,
                                         w_mean=curr_pred_mean,
                                         w_L=curr_pred_L,
                                         **curr_rec_input)

        loss = nll_loss_(true_val=curr_final_ground_truth,
                         pred_mean=curr_traj_mean,
                         pred_diag=None,
                         pred_off_diag=None,
                         pred_L=None,
                         pred_cov=curr_traj_cov)

        total_loss += loss

    total_loss /= num_replan_steps
    return total_loss


def get_loss_func_dict():
    return {"nll_loss": nll_loss,
            "mp_rec_mse_loss": mp_rec_mse_loss,
            "nmll_loss": nmll_loss,
            "mp_rec_ll_loss": mp_rec_ll_loss,
            "mp_rec_digit_mse_loss": mp_rec_digit_mse_loss,
            "mp_rec_digit_ll_loss": mp_rec_digit_ll_loss,
            "mp_rec_digit_replan_ll_loss": mp_rec_digit_replan_ll_loss
            }


def get_rec_loss_func_list():
    return {"mp_rec_ll_loss",
            "mp_rec_digit_mse_loss",
            "mp_rec_mse_loss",
            "mp_rec_digit_ll_loss",
            "mp_rec_digit_replan_ll_loss"
            }
