import copy
import csv
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from nmp.mp import *
from nmp import util
from nmp.net import MPNet
from nmp.loss import *

from test_data_generate import TrajectoryGenerator


# Check if GPU is available
util.check_torch_device()


def pap_idmp_to_traj(reconstructor_input, w_mean, w_L=None, **kwargs):
    # Get config_name
    config_name = kwargs["idmp_config"]
    tr = TrajectoriesReconstructor(config_name)

    reconstructor_input = copy.deepcopy(reconstructor_input)

    # Remove the redundant axis of num_time_group
    if reconstructor_input["bc_index"].ndim == 2:
        for key, value in reconstructor_input.items():
            v = value[:, 0]
            reconstructor_input[key] = v

    # reconstructor_input["time_indices"] = None
    reconstructor_input["time_indices"] = tr.mp.pc_times[reconstructor_input['bc_index'].long().item():]
    reconstructor_input["time_indices"] = torch.arange(reconstructor_input['bc_index'].long().item(),
                                                       len(tr.mp.pc_times), 1)
    # Remove time axis because mp weights are time-independent
    w_mean = w_mean.squeeze(-2)
    if w_L is not None:
        w_L = w_L.squeeze(-3)

    # Get time duration
    normalizer = kwargs["normalizer"]
    duration = (normalizer["time"]["max"] - normalizer["time"]["min"]).item()

    # std
    reconstructor_input["std"] = kwargs["std"]

    # Get velocity
    get_velocity = kwargs.get("get_velocity", False)
    reconstructor_input["get_velocity"] = get_velocity

    # Reconstruct
    results = tr.reconstruct(duration,
                             w_mean=w_mean,
                             w_L=w_L,  # change from w_L
                             **reconstructor_input)

    # Reshape
    num_dof = tr.get_config()["num_dof"]
    reshape_results = list()
    for result in results:
        if result is not None:
            reshape_result = result.reshape([*result.shape[:-1], num_dof,
                                             result.shape[-1] // num_dof])
            reshape_result = torch.einsum('...ij->...ji', reshape_result)
        else:
            reshape_result = None
        reshape_results.append(reshape_result)
    return reshape_results


def pap_plot():
    raise NotImplementedError


def pap_predict_test(mp_net, exp_title, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    # Predict using test dataset
    result_dict = mp_net.use_test_dataset_to_predict()
    # Loss
    rec_ll = -result_dict["loss"]
    util.print_line_title("Reconstruct likelihood: {}".format(rec_ll))
    # Prediction for reconstruction
    pred_mean = result_dict["mean"]
    pred_L = result_dict["L"]

    # Remove agg state axis, only use the latest one
    pred_mean = pred_mean[:, -1]
    pred_L = pred_L[:, -1]

    # Some other reconstruction input
    assigned_dict = result_dict["assigned_dict"]
    reconstructor_input = assigned_dict["reconstructor_input"]

    kwargs["std"] = True
    kwargs["normalizer"] = mp_net.normalizer
    kwargs["idmp_config"] = kwargs["idmp_config"]

    test_batch = result_dict["test_batch"]
    tar_true_values = test_batch["c_pos_vel_z"]["value"][:, :, 0].cpu().numpy()
    tar_true_times = test_batch["c_pos_vel_z"]["time"].cpu().numpy()

    # util.debug_plot(tar_true_times[0,:,0].cpu().numpy(), tar_true_values.cpu().numpy())

    # Compute trajectories
    traj_mean_val, traj_std_val = pap_idmp_to_traj(reconstructor_input,
                                                   pred_mean,
                                                   pred_L,
                                                   **kwargs)

    traj_mean_val = traj_mean_val.cpu().numpy()
    traj_std_val = traj_std_val.cpu().numpy()

    print(traj_mean_val.shape)
    # util.debug_plot(tar_true_times[0,:,0].cpu().numpy(), traj_mean_val.cpu().numpy())

    # Get encoder input for plotting
    original_encoder_input = assigned_dict["original_encoder_input"]
    ctx_value = original_encoder_input["ctx"]["value"]

    for i in range(len(ctx_value)):
        plt.figure()
        plt.plot(tar_true_times[0, :, 0], tar_true_values[i], alpha=0.3)
        plt.plot(tar_true_times[0, :, 0], traj_mean_val[i])
        bc_index = ctx_value[i, 0, 0].long().cpu().numpy()
        plt.axvline(tar_true_times[0, bc_index, 0], alpha=0.2)
        plt.ylim([0.4, 0.6])
        plt.show()


def pap_predict(mp_net, dict_pred_ctx, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    # set box ctx, collision ctx
    box_ctx = dict_pred_ctx['box_ctx']
    coll_ctx = dict_pred_ctx['collision_ctx']
    # set replanning ctx
    time_re_ctx = dict_pred_ctx['replanning_time_ctx']
    coll_re_ctx = dict_pred_ctx['replanning_collision_ctx']
    # generate ground truth trajs
    tg = TrajectoryGenerator(box_ctx, coll_ctx)
    tg.set_time_scope(10, 10001)
    tg.set_replanning(time_re_ctx/2, coll_re_ctx)
    traj, vel, _, traj_re, vel_re, _ = tg.generate_trajectory_option7()
    over_h, re_over_h = tg.get_override_height()
    # set predict ctx
    traj = traj[:5001:50]
    vel = vel[:5001:50]
    traj_re = traj_re[:5001:50]
    vel_re = vel_re[:5001:50]
    ctx_tensor = torch.Tensor([time_re_ctx,
                               traj[time_re_ctx, -1],
                               vel[time_re_ctx, -1],
                               coll_re_ctx])
    dict_ctx = {'ctx': {'value': ctx_tensor}}
    # predict
    w_mean, w_L = mp_net.predict(dict_obs=dict_ctx, decoder_input=None, num_mc_smp=None)
    # Remove agg state axis, only use the latest one
    w_mean = w_mean[:, -1]
    w_L = w_L[:, -1]

    # Compute trajectories
    reconstructor_input = {'bc_index': torch.Tensor([time_re_ctx]),
                           'bc_pos': torch.Tensor([[traj[time_re_ctx, -1]]]),
                           'bc_vel': torch.Tensor([[vel[time_re_ctx, -1]]]),
                           'time_indices': None}
    kwargs["std"] = True
    kwargs["normalizer"] = mp_net.normalizer
    kwargs["idmp_config"] = kwargs["idmp_config"]

    traj_mean_val, traj_std_val = pap_idmp_to_traj(reconstructor_input,
                                                   w_mean,
                                                   w_L,
                                                   **kwargs)
    traj_mean_val = traj_mean_val.cpu().numpy()
    traj_std_val = traj_std_val.cpu().numpy()

    dict_result = {'traj': traj,
                   'rep_traj': traj_re,
                   'pred_mean': traj_mean_val.squeeze(),
                   'pred_std': traj_std_val.squeeze()}
    return dict_result


def pap_post_processing(mp_net, exp_title, **kwargs):
    """
    a temp post-processing function
    Returns:
    """
    kwargs = copy.deepcopy(kwargs)
    # set box_ctx and collision_ctx
    box_ctx = 0.02
    coll_ctx = 0.15
    # set replanning ctx
    re_time_ctx = [34, 46, 56, 62]
    re_coll_ctx = [0.075, 0.125, 0.175, 0.225]
    # initialize figure
    figure = plt.figure(figsize=(16, 1.5 * len(re_time_ctx)),
                        dpi=200,
                        tight_layout=True)
    num_sub_fig_row = len(re_time_ctx)
    num_sub_fig_col = len(re_coll_ctx)
    drawn_sub_fig = 0
    for rtc in re_time_ctx:
        for rcc in re_coll_ctx:
            drawn_sub_fig += 1
            print("plot figure number:", str(drawn_sub_fig))
            plt.subplot(num_sub_fig_row, num_sub_fig_col, drawn_sub_fig)
            # plt.gca().set_title('re_time_index='+str(rtc)+' '+'re_coll_height='+str(rcc))
            # dict_pred_ctx = dict()
            dict_pred_ctx = {'box_ctx': box_ctx,
                             'collision_ctx': coll_ctx,
                             'replanning_time_ctx': rtc,
                             'replanning_collision_ctx': rcc}
            dict_result = pap_predict(mp_net, dict_pred_ctx, **kwargs)
            # plot original traj
            y = dict_result['traj'][:, 1]
            z = dict_result['traj'][:, 2]
            plt.plot(y[:rtc], z[:rtc], linestyle='-', color='g', alpha=0.3)
            plt.plot(y[rtc:], z[rtc:], linestyle='--', color='g', alpha=0.3)
            # plot replanning traj
            z_re = dict_result['rep_traj'][:, 2]
            plt.plot(y, z_re, linestyle='-', color='r', alpha=0.3)
            # plot predict mean
            z_mean_pred = dict_result['pred_mean']
            plt.plot(y[rtc:], z_mean_pred, linestyle='-', color='b', alpha=1.0)
            # plot predict std
            z_std_pred = dict_result['pred_std']
            if z_std_pred is not None:
                plt.fill_between(y[rtc:],
                                 z_mean_pred - 2 * z_std_pred,
                                 z_mean_pred + 2 * z_std_pred,
                                 color='gray',
                                 alpha=0.5,
                                 zorder=1)
            # plot others
            plt.axvline(y[rtc], alpha=0.2)
            plt.plot(0, 0.36 + 2 * box_ctx + 2 * rcc + 0.04, marker='*')
            plt.ylim([0.4, 1.0])
            if rtc != re_time_ctx[-1]:
                plt.xticks([])
            if rcc != re_coll_ctx[0]:
                plt.yticks([])
    mp_net.log_figure(figure, exp_title + ', ' + 'initial collision height: ' + str(coll_ctx))


def nmp_idmp_pap_exp(**kwargs):
    # What are my configs?
    util.print_wrap_title("NMP IDMP of PickAndPlace")
    kwargs["idmp_config"] = "IDMP_PAP_config"   # MP config
    config_path = util.get_config_path("NMP_IDMP_PAP_config")  # NN config
    # config_path = util.get_config_path("NMP_IDMP_Panda_BB_config_mse_evalutation")
    # config_path = util.get_config_path("BEST_NMP_IDMP_BB_config")

    config = util.parse_config(config_path)  # copy
    kwargs["dataset_name"] = config["dataset"]["name"]  # copy
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_PAP", mp_net.epoch)

    kwargs["y_lim"] = [0, 0.8]
    kwargs["x_lim"] = [-0.5, 0.5]

    pap_post_processing(mp_net, exp_title, **kwargs)
    # pap_predict_test(mp_net, exp_title, **kwargs)
    # pap_predict(mp_net, exp_title, **kwargs)
    # pap_post_processing(mp_net, exp_title, **kwargs)
    # idmp_predict(mp_net, exp_title, **kwargs)
    # idmp_replanning(mp_net, exp_title, **kwargs)
    # idmp_n_ctx(mp_net, exp_title, **kwargs)


def test(exp: str, restart=True):
    exp_api = dict()

    # v6 5000
    exp_api["nmp_idmp_pap"] = {
        "func": nmp_idmp_pap_exp,
        "api": "model path should be here",
        "best_epoch": 16000}

    # Specify task
    exp_func = exp_api[exp]["func"]
    exp_kwargs = \
        {"max_epoch": 10000 if restart else exp_api[exp]["best_epoch"],
         "init_epoch": 0 if restart else exp_api[exp]["best_epoch"],
         "num_mc_smp": None,
         "model_api": None if restart else exp_api[exp]["api"],
         "manual_test": False}

    # Run task
    exp_func(**exp_kwargs)


def main():
    # test(exp='nmp_idmp_pap', restart=True)
    test(exp='nmp_idmp_pap', restart=False)


if __name__ == "__main__":
    main()
