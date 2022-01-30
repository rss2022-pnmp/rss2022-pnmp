import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from test_data_generate import PolynomialGenerator, TrajectoryGenerator
from nmp import util


def test_temp_store():
    temp_store_dir = "/home/temp_dir"
    temp_x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    temp_y = np.sin(temp_x)
    plt.plot(temp_x, temp_y)
    # plt.savefig(os.path.join(temp_store_dir, 'test_fig.png'))
    plt.show()

    print(os.path.realpath(__file__))
    print(os.path.dirname(os.path.realpath(__file__)))


def interpolation(length, data_ori):
    length_ori = data_ori.shape[0]
    length_interp = length
    x_ori = np.linspace(0, length_ori - 1, length_ori)
    x_interp = np.linspace(0, length_ori - 1, length_interp)
    data_interp = np.zeros((length, data_ori.shape[1]))
    for k in range(data_ori.shape[1]):
        data_interp[:, k] = np.interp(x_interp, x_ori, data_ori[:, k])

    return data_interp


def dict_to_frame(dict_data: dict):
    df = pd.DataFrame({'index': np.linspace(0, len(dict_data['t'])-1, len(dict_data['t']), dtype=int)})
    for key, value in dict_data.items():
        if len(value.shape) < 2:
            df_value = pd.DataFrame({key: value})
            df = pd.concat([df, df_value], axis=1)
        else:
            for i in range(value.shape[1]):
                df_value = pd.DataFrame({key + '_' + str(i): value[:, i]})
                df = pd.concat([df, df_value], axis=1)
    return df


def test_spline_function():
    # box height 0.02
    # collision height from 0.02 to 0.10
    box_ctx = 0.02
    # collision_ctx = [0.02, 0.04, 0.06, 0.08, 0.10]
    # replanning_time_ctx = [15, 20, 25, 30, 35]

    collision_ctx = [0.05, 0.075, 0.10, 0.125, 0.15]
    collision_replanning_ctx = [0.05, 0.075, 0.10, 0.125, 0.15]
    replanning_time_ctx = [15, 20, 25, 30, 35]

    t = np.linspace(0, 10, 10001)

    for ctx in collision_ctx:
        tg = TrajectoryGenerator(box_ctx, ctx)
        tg.set_time_scope(t.max(), t.shape[0])
        for ctx_re in collision_replanning_ctx:
            for time_re in replanning_time_ctx:
                tg.set_replanning(time_re, ctx_re)
                traj, vel, acc, traj_re, vel_re, acc_re = tg.generate_trajectory_option7()
                # plt.plot(traj[:, 1], traj[:, 2])
                # plt.plot(traj_re[:, 1], traj_re[:, 2])
                plt.plot(t, traj_re[:, 2])
    # plt.plot(traj[:, 1], traj[:, 2])
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.68, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.63, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.58, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.53, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.48, 100), color='b')
    plt.show()


def test_robot_dataset_generation(to_dataset_name: str):
    # create corresponding dataset dir
    dataset_path = util.get_dataset_dir(to_dataset_name)
    util.remove_file_dir(dataset_path)
    os.makedirs(dataset_path)

    box_ctx = 0.02
    collision_ctx = [0.05, 0.075, 0.10, 0.125, 0.15]
    collision_replanning_ctx = [0.05, 0.075, 0.10, 0.125, 0.15]
    replanning_time_ctx = [15, 20, 25, 30, 35]
    t = np.linspace(0, 10, 10001)  # need change for different dataset
    init_c_ctx = np.zeros(t.shape[0])
    replan_c_ctx = np.zeros(t.shape[0])
    replan_time_index = np.zeros(t.shape[0])
    replan_time = np.zeros(t.shape[0])
    name_num = 0

    # option 1 -- total 125 trajs including 20 repeated trajs
    for coll_ctx in collision_ctx:
        tg = TrajectoryGenerator(box_ctx, coll_ctx)
        tg.set_time_scope(t.max(), t.shape[0])
        for coll_re_ctx in collision_replanning_ctx:
            for time_re in replanning_time_ctx:
                data_dict = dict()
                data_dict['t'] = t
                tg.set_replanning(time_re, coll_re_ctx)
                _, _, _, traj_re, vel_re, acc_re = tg.generate_trajectory_option7()
                data_dict['c_pos'] = traj_re
                data_dict['c_vel'] = vel_re
                data_dict['c_acc'] = acc_re
                init_c_ctx[:] = coll_ctx
                # data_dict['init_c_ctx'] = init_c_ctx
                replan_c_ctx[:] = coll_re_ctx
                # data_dict['replan_c_ctx'] = replan_c_ctx
                replan_time_index[:] = time_re
                # data_dict['replan_time_index'] = replan_time_index
                replan_time[:] = t[time_re]
                # data_dict['replan_time'] = replan_time
                df = dict_to_frame(data_dict)

                # compound data
                # compound c_pos and c_vel
                list_c_compound_data = list()
                for key in ['pos', 'vel']:
                    for i in [1, 2]:
                        list_c_compound_data.append(data_dict['c_' + key][:, i])
                c_compound_data = list(np.stack(list_c_compound_data, axis=-1))
                df['c_pos_vel'] = c_compound_data

                # compound c_pos_z and c_vel_Z
                list_c_compound_data = list()
                for key in ['pos', 'vel']:
                    for i in [2]:
                        list_c_compound_data.append(data_dict['c_' + key][:, i])
                c_compound_data = list(np.stack(list_c_compound_data, axis=-1))
                df['c_pos_vel_z'] = c_compound_data
                # compound ctx [re_time_index re_c_pos_1 re_c_pos_2 re_c_vel_1 re_c_vel_2 re_collision_ctx]
                list_ctx_compound_data = list()
                list_ctx_compound_data.append(np.array(time_re))
                c_pos_ctx = np.zeros((t.shape[0], 2))
                abs_index = int(time_re*(len(t)-1)/100)
                c_pos_ctx[:, 0] = data_dict['c_pos'][abs_index, 1]
                c_pos_ctx[:, 1] = data_dict['c_pos'][abs_index, 2]
                # list_ctx_compound_data.append(c_pos_ctx[:, 0][0])
                list_ctx_compound_data.append(c_pos_ctx[:, 1][0])
                c_vel_ctx = np.zeros((t.shape[0], 2))
                c_vel_ctx[:, 0] = data_dict['c_vel'][abs_index, 1]
                c_vel_ctx[:, 1] = data_dict['c_vel'][abs_index, 2]
                # list_ctx_compound_data.append(c_vel_ctx[:, 0][0])
                list_ctx_compound_data.append(c_vel_ctx[:, 1][0])
                list_ctx_compound_data.append(np.array(coll_re_ctx))
                ctx_compound_data = list(np.stack(list_ctx_compound_data, axis=-1)[None])

                # df['ctx'] = ctx_compound_data
                df_ctx = pd.DataFrame({'ctx': ctx_compound_data})
                df_ctx.to_csv(path_or_buf=dataset_path+'/static_'+str(name_num),
                              index=False,
                              quoting=csv.QUOTE_ALL)
                df.to_csv(path_or_buf=dataset_path + '/' + str(name_num),
                          index=False,
                          quoting=csv.QUOTE_ALL)
                name_num += 1
                print(name_num)


def test_plot(dataset_name):
    # read original dataset
    list_pd_df, _ = util.read_dataset(dataset_name)
    t = np.linspace(0, 10, 10001)
    for pd_df in list_pd_df:
        j_pos = pd_df['joint_vel_4'].values
        if j_pos.dtype == object:
            j_pos = util.from_string_to_array(j_pos)
        plt.plot(t[:500], j_pos[:500])
    plt.show()
    # raise NotImplementedError


if __name__ == '__main__':
    # test_temp_store()
    # test_spline_function()
    test_robot_dataset_generation('robot_pick_and_place/exp7/pap_10000_compound')
    # test_plot('pap_joint')
