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
    # reconstructor_input["time_indices"] = tr.mp.pc_times[reconstructor_input['bc_index'].long().item():]
    reconstructor_input["time_indices"] = torch.arange(reconstructor_input['bc_index'][0].long().item(),
                                                       len(tr.mp.pc_times), 1)
    reconstructor_input["time_indices"] = reconstructor_input["time_indices"].expand(
        reconstructor_input["bc_index"].shape[0], -1)
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


def panda_pap_predict_test(mp_net, exp_title, **kwargs):
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

    # mvn = MultivariateNormal(loc=pred_mean, scale_tril=pred_L, validate_args=False)
    # samples = mvn.sample([10])   #[num_smp, num_traj, ...]

    # Some other reconstruction input
    assigned_dict = result_dict["assigned_dict"]
    list_pred_mean = list()
    list_pred_std = list()
    # reconstruct
    for i in range(pred_mean.shape[0]):

        # reconstructor_input = assigned_dict["reconstructor_input"]
        reconstructor_input = dict()
        for key, value in assigned_dict["reconstructor_input"].items():
            reconstructor_input[key] = assigned_dict["reconstructor_input"][key][i:i+1]
            reconstructor_input[key] = reconstructor_input[key].expand(pred_mean.shape[0],
                                                                       *reconstructor_input[key].shape[1:])
        reconstructor_input['time_indices'] = assigned_dict["reconstructor_input"]["time_indices"][i:i+1, 0]

        kwargs["std"] = True
        kwargs["normalizer"] = mp_net.normalizer
        kwargs["idmp_config"] = kwargs["idmp_config"]

        pred_mean_i = pred_mean[i:i+1, :]
        pred_mean_i = pred_mean_i.expand(pred_mean.shape[0], *pred_mean.shape[1:])
        pred_L_i = pred_L[i:i+1, :]
        pred_L_i = pred_L_i.expand(pred_L.shape[0], *pred_L.shape[1:])

        # Compute trajectories
        traj_mean_val, traj_std_val = pap_idmp_to_traj(reconstructor_input,
                                                       pred_mean_i,
                                                       pred_L_i,
                                                       **kwargs)

        traj_mean_val = traj_mean_val.cpu().numpy()
        traj_std_val = traj_std_val.cpu().numpy()
        list_pred_mean.append(traj_mean_val[0])
        list_pred_std.append(traj_std_val[0])

    test_batch = result_dict["test_batch"]
    tar_true_values = test_batch["j_pos_vel"]["value"].cpu().numpy()
    tar_true_times = test_batch["j_pos_vel"]["time"].cpu().numpy()

    # Get encoder input for plotting
    original_encoder_input = assigned_dict["original_encoder_input"]
    ctx_value = original_encoder_input["ctx"]["value"].cpu().numpy()

    # corresponding without replanning index
    no_re_index = result_dict['test_batch']['corresponding_index']['value'].long().cpu().numpy().squeeze()
    traj_num = result_dict['test_batch']['corresponding_num']['value'].long().cpu().numpy().squeeze()

    dict_pred_test = {'tar_true_times': tar_true_times,
                      'tar_true_values': tar_true_values,
                      'ctx_value': ctx_value,
                      'without_replanning_traj_index': no_re_index,
                      'traj_num': traj_num,
                      'list_pred_mean': list_pred_mean,
                      'list_pred_std': list_pred_std}

    return dict_pred_test


def panda_pap_predict_test_sample(mp_net, exp_title, **kwargs):
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
    pred_mean = pred_mean[:, 0]
    pred_L = pred_L[:, 0]

    mvn = MultivariateNormal(loc=pred_mean, scale_tril=pred_L, validate_args=False)
    pred_sample = mvn.sample([20])   #[num_smp, num_traj, ...]

    # Some other reconstruction input
    assigned_dict = result_dict["assigned_dict"]
    list_pred_mean = list()
    list_pred_sample = list()
    list_pred_std = list()
    # reconstruct
    for i in range(pred_mean.shape[0]):
        # reconstructor_input = assigned_dict["reconstructor_input"]
        reconstructor_input = dict()
        for key, value in assigned_dict["reconstructor_input"].items():
            reconstructor_input[key] = assigned_dict["reconstructor_input"][key][i:i+1]
            reconstructor_input[key] = reconstructor_input[key].expand(pred_mean.shape[0],
                                                                       *reconstructor_input[key].shape[1:])
        reconstructor_input['time_indices'] = assigned_dict["reconstructor_input"]["time_indices"][i:i+1, 0]

        kwargs["std"] = True
        kwargs["normalizer"] = mp_net.normalizer
        kwargs["idmp_config"] = kwargs["idmp_config"]

        # predict man and std
        pred_mean_i = pred_mean[i:i+1, :]
        pred_mean_i = pred_mean_i.expand(pred_mean.shape[0], *pred_mean.shape[1:])
        pred_L_i = pred_L[i:i+1, :]
        pred_L_i = pred_L_i.expand(pred_L.shape[0], *pred_L.shape[1:])
        traj_mean_val, traj_std_val = pap_idmp_to_traj(reconstructor_input,
                                                       pred_mean_i,
                                                       pred_L_i,
                                                       **kwargs)
        traj_mean_val = traj_mean_val.cpu().numpy()
        traj_std_val = traj_std_val.cpu().numpy()
        list_pred_mean.append(traj_mean_val[0])
        list_pred_std.append(traj_std_val[0])

        # pred sample
        pred_sample_i = pred_sample[:, i, :]
        traj_sample_val, traj_std_val = pap_idmp_to_traj(reconstructor_input,
                                                              pred_sample_i,
                                                              pred_L_i,
                                                              **kwargs)

        traj_sample_val = traj_sample_val.cpu().numpy()
        list_pred_sample.append(traj_sample_val)

    test_batch = result_dict["test_batch"]
    tar_true_values = test_batch["j_pos_vel"]["value"].cpu().numpy()
    tar_true_times = test_batch["j_pos_vel"]["time"].cpu().numpy()

    # Get encoder input for plotting
    original_encoder_input = assigned_dict["original_encoder_input"]
    ctx_value = original_encoder_input["ctx"]["value"].cpu().numpy()

    # corresponding without replanning index
    no_re_index = result_dict['test_batch']['corresponding_index']['value'].long().cpu().numpy().squeeze()
    traj_num = result_dict['test_batch']['corresponding_num']['value'].long().cpu().numpy().squeeze()

    dict_pred_test = {'tar_true_times': tar_true_times,
                      'tar_true_values': tar_true_values,
                      'ctx_value': ctx_value,
                      'without_replanning_traj_index': no_re_index,
                      'traj_num': traj_num,
                      'list_pred_mean': list_pred_mean,
                      'list_pred_sample': list_pred_sample,
                      'list_pred_std': list_pred_std}

    return dict_pred_test


def panda_pap_dataset(dataset_name):
    list_pd_df, list_pd_df_static = util.read_dataset(dataset_name=dataset_name)
    t = list_pd_df[0]["t"]
    num_dof = 7
    list_j_pos = list()
    list_j_pos_without_replanning = list()
    for i in range(len(list_pd_df)):
        j_pos = np.zeros([t.shape[0], num_dof])
        for dof in range(num_dof):
            j_pos[:, dof] = list_pd_df[i]['j_pos_'+str(dof)]
        list_j_pos.append(j_pos)
        if i in [0, 30, 60, 90, 120]:
            list_j_pos_without_replanning.append(j_pos)
    dict_ori_dataset = dict()
    dict_ori_dataset['list_joint_position'] = list_j_pos
    dict_ori_dataset['list_joint_position_without_replanning'] = list_j_pos_without_replanning
    return dict_ori_dataset


def panda_pap_post_processing(mp_net, exp_title, **kwargs):
    # parse dict
    dict_pred_test = panda_pap_predict_test(mp_net, exp_title, **kwargs)
    plot_t = dict_pred_test['tar_true_times'][0]  # [101 1]
    plot_target = dict_pred_test['tar_true_values'][:, :, :7]  # [10 101 7]
    plot_ctx = dict_pred_test["ctx_value"]  # [10 1 4]
    list_plot_pred_mean = dict_pred_test['list_pred_mean']  # [10][101-ctx[..., 0] 7]
    list_plot_pred_std = dict_pred_test['list_pred_std']  # [10][101-ctx[..., 0] 7]
    plot_no_re_index = dict_pred_test['without_replanning_traj_index']/30
    plot_traj_num = dict_pred_test['traj_num']

    print(plot_no_re_index)
    print(plot_traj_num)

    # original dataset
    dataset_name = kwargs["dataset_name"]
    dict_ori_dataset = panda_pap_dataset(dataset_name)
    list_j_pos = dict_ori_dataset['list_joint_position']  # [125][101 7]
    list_j_pos_no_re = dict_ori_dataset['list_joint_position_without_replanning']  # [5][101 7]

    # set parameters
    # plot_index = [4, 2, 6]  # [7, 4, 3, 2, 1, 6, 8] [7, 4, 2, 1, 6]
    plot_index = [0, 14, 6, 12, 16]
    list_y_lim = [[-0.6, 0.1], [-0.6, 0.4], [-0.6, 0.1], [-2.8, -2.1], [-0.2, 0.1], [1.5, 3.0], [0.0, 0.9]]
    num_dof = 7
    figure = plt.figure(figsize=(24, 1.5 * num_dof),
                        dpi=200,
                        tight_layout=True)
    num_sub_fig_row = num_dof
    num_sub_fig_col = len(plot_index) + 1
    drawn_sub_fig = 0

    # plot raw dataset todo
    drawn_sub_fig += 1
    for dof in range(num_dof):
        plt.subplot(num_sub_fig_row, num_sub_fig_col, drawn_sub_fig + num_sub_fig_col * dof)
        plt.ylim(list_y_lim[dof])
        for i in range(len(list_j_pos)):
            plt.plot(plot_t, list_j_pos[i][:, dof], linewidth=0.5)

    # plot individual trajs
    for index in plot_index:
        drawn_sub_fig += 1
        for dof in range(num_dof):
            plt.subplot(num_sub_fig_row, num_sub_fig_col, drawn_sub_fig + num_sub_fig_col * dof)
            print('plot index: ', drawn_sub_fig + 4 * dof)
            bc_index = int(plot_ctx[index, :, 0])
            # plot traj without replanning
            no_re_index = int(plot_no_re_index[index])
            plt.plot(plot_t[:bc_index], list_j_pos_no_re[no_re_index][:bc_index, dof],
                     linestyle='-', color='g', alpha=0.3)
            plt.plot(plot_t[bc_index:], list_j_pos_no_re[no_re_index][bc_index:, dof],
                     linestyle='--', color='g', alpha=0.3)
            # plot ground truth
            plt.plot(plot_t, plot_target[index, :, dof],
                     linestyle='-', color='r', alpha=0.3)  # traj true dimension?
            # plot predict
            plt.plot(plot_t[bc_index:], list_plot_pred_mean[index][:, dof],
                     linestyle='-', color='b', alpha=1.0)
            plt.fill_between(plot_t[bc_index:, 0],
                             list_plot_pred_mean[index][:, dof] - 2 * list_plot_pred_std[index][:, dof],
                             list_plot_pred_mean[index][:, dof] + 2 * list_plot_pred_std[index][:, dof],
                             color='gray',
                             alpha=0.5,
                             zorder=1)
            # plot others
            plt.axvline(plot_t[bc_index], alpha=0.2)
            plt.plot(plot_t[-1, 0], plot_target[index][-1, dof], marker='*')
            plt.ylim(list_y_lim[dof])
            plt.yticks([])
        #   plt.show()
    mp_net.log_figure(figure, exp_title)


def panda_pap_sample_post_processing(mp_net, exp_title, **kwargs):
    # parse dict
    dict_pred_test = panda_pap_predict_test_sample(mp_net, exp_title, **kwargs)
    plot_t = dict_pred_test['tar_true_times'][0]  # [101 1]
    plot_target = dict_pred_test['tar_true_values'][:, :, :7]  # [20 101 7]
    plot_ctx = dict_pred_test["ctx_value"]  # [20 1 4]
    list_plot_pred_mean = dict_pred_test['list_pred_mean']  # [101-ctx[..., 0] 7]
    list_plot_pred_sample = dict_pred_test['list_pred_sample']  # [20][20 101-ctx[..., 0] 7]
    list_plot_pred_std = dict_pred_test['list_pred_std']  # [101-ctx[..., 0] 7]
    plot_no_re_index = dict_pred_test['without_replanning_traj_index']/30
    plot_traj_num = dict_pred_test['traj_num']

    print(plot_no_re_index)
    print(plot_traj_num)

    # original dataset
    dataset_name = kwargs["dataset_name"]
    dict_ori_dataset = panda_pap_dataset(dataset_name)
    list_j_pos = dict_ori_dataset['list_joint_position']  # [125][101 7]
    list_j_pos_no_re = dict_ori_dataset['list_joint_position_without_replanning']  # [5][101 7]

    # set parameters
    # plot_index = [4, 2, 6]  # [7, 4, 3, 2, 1, 6, 8] [7, 4, 2, 1, 6]
    plot_index = [0, 14, 6, 12, 16]
    list_y_lim = [[-0.6, 0.1], [-0.6, 0.4], [-0.6, 0.1], [-2.8, -2.1], [-0.2, 0.1], [1.5, 3.0], [0.0, 0.9]]
    num_dof = 7
    figure = plt.figure(figsize=(24, 1.5 * num_dof),
                        dpi=200,
                        tight_layout=True)
    num_sub_fig_row = num_dof
    num_sub_fig_col = len(plot_index) + 1
    drawn_sub_fig = 0

    # plot raw dataset todo
    drawn_sub_fig += 1
    for dof in range(num_dof):
        plt.subplot(num_sub_fig_row, num_sub_fig_col, drawn_sub_fig + num_sub_fig_col * dof)
        plt.ylim(list_y_lim[dof])
        for i in range(len(list_j_pos)):
            plt.plot(plot_t, list_j_pos[i][:, dof], linewidth=0.5)

    # plot individual trajs
    for index in plot_index:
        drawn_sub_fig += 1
        for dof in range(num_dof):
            plt.subplot(num_sub_fig_row, num_sub_fig_col, drawn_sub_fig + num_sub_fig_col * dof)
            print('plot index: ', drawn_sub_fig + 4 * dof)
            bc_index = int(plot_ctx[index, :, 0])
            # plot traj without replanning
            no_re_index = int(plot_no_re_index[index])
            plt.plot(plot_t[:bc_index], list_j_pos_no_re[no_re_index][:bc_index, dof],
                     linestyle='-', color='g', alpha=0.3)
            plt.plot(plot_t[bc_index:], list_j_pos_no_re[no_re_index][bc_index:, dof],
                     linestyle='--', color='g', alpha=0.3)
            # plot ground truth
            plt.plot(plot_t, plot_target[index, :, dof],
                     linestyle='-', color='r', alpha=0.3)  # traj true dimension?
            # plot predict
            for s in range(0, len(list_plot_pred_sample), 2):
                plt.plot(plot_t[bc_index:], list_plot_pred_sample[index][s, :, dof],
                         linestyle='-', color='b', alpha=0.3)
            plt.plot(plot_t[bc_index:], list_plot_pred_mean[index][:, dof],
                     linestyle='-', color='b', alpha=1.0)
            plt.fill_between(plot_t[bc_index:, 0],
                             list_plot_pred_mean[index][:, dof] - 2 * list_plot_pred_std[index][:, dof],
                             list_plot_pred_mean[index][:, dof] + 2 * list_plot_pred_std[index][:, dof],
                             color='gray',
                             alpha=0.5,
                             zorder=1)
            # plot others
            plt.axvline(plot_t[bc_index], alpha=0.2)
            plt.plot(plot_t[-1, 0], plot_target[index][-1, dof], marker='*')
            plt.ylim(list_y_lim[dof])
            plt.yticks([])
        #   plt.show()
    mp_net.log_figure(figure, exp_title)


def nmp_idmp_panda_pap_exp(**kwargs):
    # What are my configs?
    util.print_wrap_title("NMP IDMP of PickAndPlace")
    kwargs["idmp_config"] = "IDMP_Panda_PAP_config"   # MP config
    config_path = util.get_config_path("NMP_IDMP_Panda_PAP_config")  # NN config
    # config_path = util.get_config_path("NMP_IDMP_Panda_BB_config_mse_evalutation")
    # config_path = util.get_config_path("BEST_NMP_IDMP_BB_config")

    config = util.parse_config(config_path)  # copy
    kwargs["dataset_name"] = config["dataset"]["name"]  # copy
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    exp_title = "{}, Epoch: {}".format("NMP_IDMP_Panda_PAP", mp_net.epoch)

    kwargs["y_lim"] = [0, 0.8]
    kwargs["x_lim"] = [-0.5, 0.5]

    # panda_pap_post_processing(mp_net, exp_title, **kwargs)
    panda_pap_sample_post_processing(mp_net, exp_title, **kwargs)
    # pap_predict_test(mp_net, exp_title, **kwargs)
    # pap_predict(mp_net, exp_title, **kwargs)
    # pap_post_processing(mp_net, exp_title, **kwargs)
    # idmp_predict(mp_net, exp_title, **kwargs)
    # idmp_replanning(mp_net, exp_title, **kwargs)
    # idmp_n_ctx(mp_net, exp_title, **kwargs)


def test(exp: str, restart=True):
    exp_api = dict()

    # 7idmp v8 5500
    # 15idmp
    exp_api["nmp_idmp_panda_pap"] = {
        "func": nmp_idmp_panda_pap_exp,
        "api": "artifact = model path should be here",
        "best_epoch": 10000}

    # Specify task
    exp_func = exp_api[exp]["func"]
    exp_kwargs = \
        {"max_epoch": 20000 if restart else exp_api[exp]["best_epoch"],
         "init_epoch": 0 if restart else exp_api[exp]["best_epoch"],
         "num_mc_smp": None,
         "model_api": None if restart else exp_api[exp]["api"],
         "manual_test": False}

    # Run task
    exp_func(**exp_kwargs)


def main():
    # test(exp='nmp_idmp_panda_pap', restart=True)
    test(exp='nmp_idmp_panda_pap', restart=False)


if __name__ == "__main__":
    main()
