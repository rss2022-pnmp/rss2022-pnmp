import os
import torch

import matplotlib.pyplot as plt

from nmp import util
from nmp.net import MPNet
from nmp.data_process import BatchProcess
from test_data_generate import TrajectoryGenerator
import exp_nidmp_pap
import exp_nidmp_panda_pap

# Check if GPU is available
util.check_torch_device()


def get_tmp_dir():
    """
    Get the path to the tmp folder
    Args:
    Returns:
        path to the directory storing the dataset
    """
    tmp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "tmp_fig")
    return tmp_path


def get_model(**kwargs):
    # What are my configs?
    util.print_wrap_title(kwargs["title"])
    config_path = util.get_config_path(kwargs['NN_config'])  # NN config
    config = util.parse_config(config_path)  # copy
    kwargs["dataset_name"] = config["dataset"]["name"]  # copy
    mp_net = MPNet(config,
                   kwargs["max_epoch"],
                   kwargs["init_epoch"],
                   kwargs["model_api"])
    mp_net.fit()
    return mp_net


def plot_demo():
    # plt.figure(figsize=[8, 3])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 3)
    fig.set_dpi(200)

    # set initial box and collision ctx
    box_ctx = 0.02
    coll_ctx = 0.10
    # generate and plot ground truth trajs
    tg = TrajectoryGenerator(box_ctx, coll_ctx)
    tg.set_time_scope(10, 10001)
    tg.set_replanning(0, coll_ctx)
    traj, _, _, _, _, _ = tg.generate_trajectory_option7()
    traj = traj[:5001:50]
    ori, = ax.plot(traj[:, 1]+0.3, traj[:, 2])
    list_marker = ['*', 's', 'd', '^']
    # generate and plot replanning ctx and traj
    re_time_ctx = [40, 50, 60, 70]
    re_coll_ctx = [0.04, 0.07, 0.16, 0.13]
    for i, (rtc, rcc) in enumerate(zip(re_time_ctx, re_coll_ctx)):
        print(rtc, rcc)
        tg = TrajectoryGenerator(box_ctx, coll_ctx)
        tg.set_time_scope(10, 10001)
        tg.set_replanning(int(rtc/2), rcc)
        override_height = tg.get_override_height()
        _, _, _, traj_re, _, _ = tg.generate_trajectory_option7()
        traj_re = traj_re[:5001:50]
        # plot replanning traj
        re, = ax.plot(traj_re[rtc:, 1]+0.3, traj_re[rtc:, 2], linestyle='--', color='g')
        # plot rtc and rcc
        rt = ax.axvline(traj_re[rtc, 1]+0.3, alpha=0.8)
        ax.plot(traj[-1, 1] + 0.3, traj[-1, 2], marker='o')
        ax.plot(traj_re[-1, 1]+0.3, traj_re[-1, 2], marker=list_marker[i])
    ax.set_ylim([0.4, 0.80])
    ax.set_xticklabels([0.0, 0.1, 0.2, 0.3], fontsize=12)
    ax.set_yticklabels([0.4, 0.5, 0.6, 0.7, 0.8], fontsize=12)
    tmp_path = get_tmp_dir()
    ax.legend([ori, re, rt], ['original trajectory', 'replanning trajectory', 'replanning position'])
    plt.savefig(os.path.join(tmp_path, "replanning_demo.pdf"),
                bbox_inches="tight")
    # raise NotImplementedError


def plot_dataset():
    plt.figure(figsize=[8, 3])

    list_pd_df, _ = util.read_dataset('pap_100')
    for pd_df in list_pd_df:
        y = pd_df['c_pos_1']
        z = pd_df['c_pos_2']
        plt.plot(y+0.3, z, linewidth=0.5)
    tmp_path = get_tmp_dir()
    plt.ylim([0.4, 0.80])
    plt.savefig(os.path.join(tmp_path, "dataset.pdf"),
                bbox_inches="tight")


def plot_cnmp_task_space():
    # plt.figure(figsize=[8, 3])
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    fig.set_dpi(200)
    # get mp model
    model_kwargs = \
        {"max_epoch": 20000,
         "init_epoch": 20000,
         "num_mc_smp": None,
         "model_api": "model should be here",
         "manual_test": False,
         "title": "CNMP of PickAndPlace",
         'NN_config': "CNMP_PAP_config"}
    mp_net = get_model(**model_kwargs)

    # get traj without replanning
    # set initial box and collision ctx
    box_ctx = 0.02
    coll_ctx = 0.10
    # generate and plot ground truth trajs
    tg = TrajectoryGenerator(box_ctx, coll_ctx)
    tg.set_time_scope(10, 10001)
    tg.set_replanning(0, coll_ctx)
    traj, vel, _, _, _, _ = tg.generate_trajectory_option7()
    traj = traj[:5001:50]
    vel = vel[:5001:50]
    ori_traj, = ax.plot(traj[:, 1] + 0.3, traj[:, 2])
    # get ground truth
    re_time_ctx = [33, 67]
    re_coll_ctx = [0.04, 0.16]
    for rtc, rcc in zip(re_time_ctx, re_coll_ctx):
        print(rtc, rcc)
        tg = TrajectoryGenerator(box_ctx, coll_ctx)
        tg.set_time_scope(10, 10001)
        tg.set_replanning(int(rtc/2), rcc)
        override_height = tg.get_override_height()
        _, _, _, traj_re, _, _ = tg.generate_trajectory_option7()
        traj_re = traj_re[:5001:50]
        # plot replanning traj
        re_traj, = ax.plot(traj_re[rtc:, 1]+0.3, traj_re[rtc:, 2], linestyle='--', color='g', alpha=0.3)

        # set predict ctx and decoder_input
        ctx_tensor = torch.Tensor([rtc,
                                   traj[rtc, -1],
                                   vel[rtc, -1],
                                   rcc])
        encoder_input = {'ctx': {'value': ctx_tensor}}
        time = torch.linspace(0, 5, 101)
        dict_decoder_input = {'c_pos_2': {'time': time[None, rtc:, None]}}
        norm_dict_decoder_input = BatchProcess.batch_normalize(dict_decoder_input, mp_net.normalizer)
        norm_decoder_input = norm_dict_decoder_input['c_pos_2']['time']
        # predict
        pred_mean, pred_L = mp_net.predict(dict_obs=encoder_input, decoder_input=norm_decoder_input, num_mc_smp=None)
        # Remove agg state axis, only use the latest one
        pred_mean = pred_mean[:, -1].cpu().numpy().squeeze()
        pred_L = pred_L[:, -1].cpu().numpy().squeeze()
        pred_traj, = ax.plot(traj_re[rtc:, 1]+0.3, pred_mean, linestyle='--', color='r', alpha=1.0)
        ax.fill_between(traj_re[rtc:, 1]+0.3,
                         pred_mean - 2 * pred_L,
                         pred_mean + 2 * pred_L,
                         color='gray',
                         alpha=0.5,
                         zorder=1)

        # plot rtc and rcc
        # c = plt.Circle((traj_re[rtc, 1]+0.3, traj_re[rtc, 2]), 0.02, color='y', fill=False)
        # plt.add_patch(c)
        ax.axvline(traj_re[rtc, 1]+0.3, alpha=0.2)
        ax.plot(traj_re[-1, 1]+0.3, traj_re[-1, 2], marker='*')
    tmp_path = get_tmp_dir()
    ax.set_xlim([0.09, 0.31])
    ax.set_ylim([0.54, 0.77])

    # plot a circle
    ax.scatter(traj[rtc, 1]+0.3, traj[rtc, 2], marker='o', s=150, facecolors='None', edgecolors='r')
    # plot subplot
    a = plt.axes([0.18, 0.29, 0.15, 0.3])
    plt.plot(traj[:, 1] + 0.3, traj[:, 2])
    plt.plot(traj_re[rtc:, 1] + 0.3, pred_mean, linestyle='--', color='r', alpha=1.0)
    plt.fill_between(traj_re[rtc:, 1] + 0.3,
                     pred_mean - 2 * pred_L,
                     pred_mean + 2 * pred_L,
                     color='gray',
                     alpha=0.5,
                     zorder=1)
    plt.ylim([0.56, 0.58])
    plt.xlim([0.12, 0.14])
    plt.xticks([0.0, 0.1, 0.2, 0.3], fontsize=12)
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8], fontsize=12)

    ax.legend([ori_traj, re_traj, pred_traj], ['original trajectory', 'ground truth', 'predict'])
    plt.savefig(os.path.join(tmp_path, "cnmp_replanning_demo.pdf"),
                bbox_inches="tight")

    # raise NotImplementedError


def plot_nidmp_task_space():
    # plt.figure(figsize=[8, 3])
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    fig.set_dpi(200)
    # get mp model
    model_kwargs = \
        {"max_epoch": 6000,
         "init_epoch": 6000,
         "num_mc_smp": None,
         "model_api": "model should be here",
         "manual_test": False,
         "title": "NMP IDMP of PickAndPlace",
         'idmp_config': "IDMP_PAP_config",
         'NN_config': "NMP_IDMP_PAP_config"}
    mp_net = get_model(**model_kwargs)

    # get traj without replanning
    # set initial box and collision ctx
    box_ctx = 0.02
    coll_ctx = 0.10
    # generate and plot ground truth trajs
    tg = TrajectoryGenerator(box_ctx, coll_ctx)
    tg.set_time_scope(10, 10001)
    tg.set_replanning(0, coll_ctx)
    traj, vel, _, _, _, _ = tg.generate_trajectory_option7()
    traj = traj[:5001:50]
    vel = vel[:5001:50]
    ori_traj, = ax.plot(traj[:, 1] + 0.3, traj[:, 2])

    # get ground truth
    re_time_ctx = [33, 67]
    re_coll_ctx = [0.04, 0.16]
    for rtc, rcc in zip(re_time_ctx, re_coll_ctx):
        print(rtc, rcc)
        tg = TrajectoryGenerator(box_ctx, coll_ctx)
        tg.set_time_scope(10, 10001)
        tg.set_replanning(int(rtc/2), rcc)
        override_height = tg.get_override_height()
        _, _, _, traj_re, _, _ = tg.generate_trajectory_option7()
        traj_re = traj_re[:5001:50]
        # plot replanning traj
        re_traj, = ax.plot(traj_re[rtc:, 1]+0.3, traj_re[rtc:, 2], linestyle='--', color='g', alpha=0.3)

        # set predict ctx and decoder_input
        ctx_tensor = torch.Tensor([rtc,
                                   traj[rtc, -1],
                                   vel[rtc, -1],
                                   rcc])
        dict_ctx = {'ctx': {'value': ctx_tensor}}
        # predict
        w_mean, w_L = mp_net.predict(dict_obs=dict_ctx, decoder_input=None, num_mc_smp=None)
        # Remove agg state axis, only use the latest one
        w_mean = w_mean[:, -1]
        w_L = w_L[:, -1]
        reconstructor_input = {'bc_index': torch.Tensor([rtc]),
                               'bc_pos': torch.Tensor([[traj[rtc, -1]]]),
                               'bc_vel': torch.Tensor([[vel[rtc, -1]]]),
                               'time_indices': None}
        model_kwargs["std"] = True
        model_kwargs["normalizer"] = mp_net.normalizer

        traj_mean_val, traj_std_val = exp_nidmp_pap.pap_idmp_to_traj(reconstructor_input,
                                                                     w_mean,
                                                                     w_L,
                                                                     **model_kwargs)
        pred_mean = traj_mean_val.cpu().numpy().squeeze()
        pred_L = traj_std_val.cpu().numpy().squeeze()

        pred_traj, = ax.plot(traj_re[rtc:, 1]+0.3, pred_mean, linestyle='--', color='r', alpha=1.0)
        ax.fill_between(traj_re[rtc:, 1]+0.3,
                         pred_mean - 2 * pred_L,
                         pred_mean + 2 * pred_L,
                         color='gray',
                         alpha=0.5,
                         zorder=1)

        # plot rtc and rcc
        plt.axvline(traj_re[rtc, 1]+0.3, alpha=0.2)
        plt.plot(traj_re[-1, 1]+0.3, traj_re[-1, 2], marker='*')
    tmp_path = get_tmp_dir()
    ax.set_xlim([0.09, 0.31])
    ax.set_ylim([0.54, 0.77])

    # plot a circle
    ax.scatter(traj[rtc, 1] + 0.3, traj[rtc, 2], marker='o', s=150, facecolors='None', edgecolors='r')
    # plot subplot
    a = plt.axes([0.18, 0.29, 0.15, 0.3])
    plt.plot(traj[:, 1] + 0.3, traj[:, 2])
    plt.plot(traj_re[rtc:, 1] + 0.3, pred_mean, linestyle='--', color='r', alpha=1.0)
    plt.fill_between(traj_re[rtc:, 1] + 0.3,
                     pred_mean - 2 * pred_L,
                     pred_mean + 2 * pred_L,
                     color='gray',
                     alpha=0.5,
                     zorder=1)
    plt.ylim([0.56, 0.58])
    plt.xlim([0.12, 0.14])
    plt.xticks([])
    plt.yticks([])

    ax.legend([ori_traj, re_traj, pred_traj], ['original trajectory', 'ground truth', 'predict'])
    plt.savefig(os.path.join(tmp_path, "nmp_idmp_replanning_demo.pdf"),
                bbox_inches="tight")
    # raise NotImplementedError


def plot_nidmp_joint_space():
    plt.figure(figsize=[8, 3])
    # get mp model
    model_kwargs = \
        {"max_epoch": 10000,
         "init_epoch": 10000,
         "num_mc_smp": None,
         "model_api": "model should be here",
         "manual_test": False,
         "title": "NMP IDMP of PickAndPlace",
         'idmp_config': "IDMP_Panda_PAP_config",
         'NN_config': "NMP_IDMP_Panda_PAP_config"}
    mp_net = get_model(**model_kwargs)
    dict_pred_test = exp_nidmp_panda_pap.panda_pap_predict_test(mp_net, model_kwargs["title"], **model_kwargs)
    plot_t = dict_pred_test['tar_true_times'][0]  # [101 1]
    plot_target = dict_pred_test['tar_true_values'][:, :, :7]  # [10 101 7]
    plot_ctx = dict_pred_test["ctx_value"]  # [10 1 4]
    list_plot_pred_mean = dict_pred_test['list_pred_mean']  # [10][101-ctx[..., 0] 7]
    list_plot_pred_std = dict_pred_test['list_pred_std']  # [10][101-ctx[..., 0] 7]
    plot_no_re_index = dict_pred_test['without_replanning_traj_index']/30
    plot_traj_num = dict_pred_test['traj_num']

    # original dataset
    dict_ori_dataset = exp_nidmp_panda_pap.panda_pap_dataset('panda_pap_100_7_idmp')
    list_j_pos = dict_ori_dataset['list_joint_position']  # [125][101 7]
    list_j_pos_no_re = dict_ori_dataset['list_joint_position_without_replanning']  # [5][101 7]

    # set parameters
    # plot_index = [4, 2, 6]  # [7, 4, 3, 2, 1, 6, 8] [7, 4, 2, 1, 6]
    plot_index = [0, 14, 6, 12, 16]
    list_y_lim = [[-0.6, 0.1], [-0.6, 0.4], [-0.6, 0.1], [-2.8, -2.1], [-0.2, 0.1], [1.5, 3.0], [0.0, 0.9]]

    # plot index
    index = 0
    dof = 3
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
    tmp_path = get_tmp_dir()
    plt.savefig(os.path.join(tmp_path, "nmp_idmp_panda_replanning_demo.pdf"),
                bbox_inches="tight")
    # raise NotImplementedError


def plot_nidmp_sample_joint_space():
    # plt.figure(figsize=[4 * 3, 1.5 * 3])
    # get mp model
    model_kwargs = \
        {"max_epoch": 10000,
         "init_epoch": 10000,
         "num_mc_smp": None,
         "model_api": "model should be here",
         "manual_test": False,
         "title": "NMP IDMP of PickAndPlace",
         'idmp_config': "IDMP_Panda_PAP_config",
         'NN_config': "NMP_IDMP_Panda_PAP_config"}
    mp_net = get_model(**model_kwargs)
    dict_pred_test = exp_nidmp_panda_pap.panda_pap_predict_test_sample(mp_net, model_kwargs["title"], **model_kwargs)
    plot_t = dict_pred_test['tar_true_times'][0]  # [101 1]
    plot_target = dict_pred_test['tar_true_values'][:, :, :7]  # [20 101 7]
    plot_ctx = dict_pred_test["ctx_value"]  # [20 1 4]
    list_plot_pred_mean = dict_pred_test['list_pred_mean']  # [101-ctx[..., 0] 7]
    list_plot_pred_sample = dict_pred_test['list_pred_sample']  # [20][20 101-ctx[..., 0] 7]
    list_plot_pred_std = dict_pred_test['list_pred_std']  # [101-ctx[..., 0] 7]
    plot_no_re_index = dict_pred_test['without_replanning_traj_index']/30
    plot_traj_num = dict_pred_test['traj_num']

    dict_ori_dataset = exp_nidmp_panda_pap.panda_pap_dataset('panda_pap_100_7_idmp')
    list_j_pos = dict_ori_dataset['list_joint_position']  # [125][101 7]
    list_j_pos_no_re = dict_ori_dataset['list_joint_position_without_replanning']  # [5][101 7]

    # set parameters
    # plot_index = [4, 2, 6]  # [7, 4, 3, 2, 1, 6, 8] [7, 4, 2, 1, 6]
    # plot_index = [0, 14, 6, 12, 16]
    plot_index = [14, 6, 12]
    list_y_lim = [[-0.6, 0.1], [-0.6, 0.4], [-0.6, 0.1], [-2.8, -2.1], [-0.2, 0.1], [1.5, 3.0], [0.0, 0.9]]
    plot_dof = [0, 3, 5]
    num_dof = 3

    # plot raw dataset
    num_sub_fig_row = len(plot_dof)
    num_sub_fig_col = len(plot_index)
    drawn_sub_fig = 0

    fig, axes = plt.subplots(num_sub_fig_row, num_sub_fig_col, sharex=True, sharey='row', squeeze=False)
    fig.set_size_inches(3.8 * num_sub_fig_col, 1.5 * num_sub_fig_row)
    fig.set_dpi(200)
    fig.tight_layout()

    # # plot raw dataset
    # drawn_sub_fig += 1
    # for d, dof in enumerate(plot_dof):
    #     ax = axes[d, 0]
    #     ax.set_ylim(list_y_lim[dof])
    #     ax.grid()
    #     for i in range(len(list_j_pos)):
    #         ax.plot(plot_t, list_j_pos[i][:, dof], linewidth=0.3)

    # plot
    # plot individual trajs
    for j, index in enumerate(plot_index):
        drawn_sub_fig += 1
        for i, dof in enumerate(plot_dof):
            ax = axes[i, j]
            # plt.subplot(num_sub_fig_row, num_sub_fig_col, drawn_sub_fig + num_sub_fig_col * i)
            print('plot index: ', drawn_sub_fig + 4 * i)
            bc_index = int(plot_ctx[index, :, 0])
            # plot traj without replanning
            no_re_index = int(plot_no_re_index[index])
            ax.plot(plot_t, list_j_pos_no_re[no_re_index][:, dof],
                    linestyle='-', alpha=0.5)
            # ax.plot(plot_t[bc_index:], list_j_pos_no_re[no_re_index][bc_index:, dof],
            #         linestyle='-', alpha=0.5)
            # plot ground truth
            ax.plot(plot_t[bc_index:], plot_target[index, bc_index:, dof],
                    linestyle='--', color='g', alpha=0.3)  # traj true dimension?
            # plot predict
            # for s in range(0, len(list_plot_pred_sample), 2):
            #     ax.plot(plot_t[bc_index:], list_plot_pred_sample[index][s, :, dof],
            #             linestyle='-', color='r', alpha=0.3)
            ax.plot(plot_t[bc_index:], list_plot_pred_mean[index][:, dof],
                    linestyle='-', color='r', alpha=1.0)
            ax.fill_between(plot_t[bc_index:, 0],
                            list_plot_pred_mean[index][:, dof] - 2 * list_plot_pred_std[index][:, dof],
                            list_plot_pred_mean[index][:, dof] + 2 * list_plot_pred_std[index][:, dof],
                            color='gray',
                            alpha=0.5,
                            zorder=1)
            # plot others
            ax.axvline(plot_t[bc_index], alpha=0.5)
            ax.plot(plot_t[-1, 0], plot_target[index][-1, dof], marker='*')
            ax.set_ylim(list_y_lim[dof])
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            ax.grid()
    tmp_path = get_tmp_dir()
    plt.savefig(os.path.join(tmp_path, "nmp_idmp_panda_sample_replanning_demo.pdf"),
                bbox_inches="tight")
    # raise NotImplementedError


if __name__ == '__main__':
    plot_demo()
    # plot_dataset()
    # plot_cnmp_task_space()
    # plot_nidmp_task_space()
    # plot_nidmp_joint_space()
    # plot_nidmp_sample_joint_space()
