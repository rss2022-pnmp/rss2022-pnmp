import os
import csv
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt

from nmp import util
from scipy.interpolate import make_interp_spline
from test_util import dict_to_frame


class PolynomialGenerator:
    def __init__(self, order: int):
        self.order = order
        self.weights = np.zeros(order+1)

    def set_weights(self, weights: list):
        self.weights = np.array(weights)

    def compute_weights(self, condition: dict):
        # solve linear equation system a @ weights = b
        a = np.ones([self.order+1, self.order+1])
        b = np.ones(self.order+1)
        condition_num = 0
        for key, value in condition.items():
            if key == 'pos_condition':
                for i in range(value.shape[0]):
                    b[condition_num] = value[i, 1]
                    for o in range(self.order):
                        a[condition_num, o] = value[i, 0]**(self.order-o)
                    condition_num += 1
            elif key == 'vel_condition':
                for i in range(value.shape[0]):
                    b[condition_num] = value[i, 1]
                    for o in range(self.order):
                        a[condition_num, o] = (self.order-o)*value[i, 0]**(self.order-o-1)
                    a[condition_num, self.order] = 0
                    condition_num += 1
            elif key == 'acc_condition':
                for i in range(value.shape[0]):
                    b[condition_num] = value[i, 1]
                    for o in range(self.order-1):
                        a[condition_num, o] = (self.order-o)*(self.order-o-1)*value[i, 0]**(self.order-o-2)
                    a[condition_num, self.order-1] = 0
                    a[condition_num, self.order] = 0
                    condition_num += 1
        # print('number of condition:', condition_num)
        self.weights = np.linalg.solve(a, b)
        if not np.allclose(np.dot(a, self.weights), b):
            raise ValueError('weights are not true')

    def compute_output_(self, x: float):
        y = 0.0
        for i in range(self.order):
            y += self.weights[i] * x ** (self.order - i)
        y += self.weights[self.order]
        return y

    def compute_output(self, x: np.ndarray):
        y = np.zeros(x.shape[-1])
        for i in range(self.order):
            y += self.weights[i] * x**(self.order-i)
        y += self.weights[self.order]
        return y

    def compute_derivative_(self, x: float):
        y_d = 0.0
        for i in range(self.order):
            y_d += (self.order-i) * self.weights[i] * x**(self.order-i-1)
        return y_d

    def compute_derivative(self, x: np.ndarray):
        y_d = np.zeros((x.shape[-1]))
        for i in range(self.order):
            y_d += (self.order-i) * self.weights[i] * x**(self.order-i-1)
        return y_d

    def compute_two_order_derivative_(self, x: float):
        y_d = 0.0
        for i in range(self.order-1):
            y_d += (self.order-i) * (self.order-i-1) * self.weights[i] * x**(self.order-i-2)
        return y_d

    def compute_two_order_derivative(self, x: np.ndarray):
        y_d = np.zeros((x.shape[-1]))
        for i in range(self.order-1):
            y_d += (self.order-i) * (self.order-i-1) * self.weights[i] * x**(self.order-i-2)
        return y_d

    def print_function(self):
        print("y = ", end=' ')
        for i in range(self.order):
            print('%s*x^%s' % (self.weights[i], (self.order-i)), '+',  end=' ')
        print(self.weights[self.order])


# class SplineGenerator:
#     raise NotImplementedError


# using 4 order for y-axis
# using 5 order for z-axis
class TrajectoryGenerator:
    def __init__(self, box_height: float, initial_collision_height: float):
        self.box_height = box_height
        self.initial_collision_height = initial_collision_height
        self.current_collision_height = self.initial_collision_height
        self.replanning_collision_height = self.initial_collision_height

        self.pick_height = 0.36 + 2 * self.box_height
        self.override_height = self.pick_height + 2 * self.initial_collision_height + 0.04
        self.replanning_override_height = self.pick_height + 2 * self.replanning_collision_height + 0.04

        # time scope
        self.time = np.linspace(0, 1.0, 101)
        self.replanning_time = 0.0
        self.replanning_time_index = 0
        self.middle_time_index = int(*np.where(self.time == np.median(self.time)))

        # initial and end position
        self.initial_pos = np.array([0.4, -0.3, self.pick_height])
        self.middle_pos = np.array([0.4, 0.0, self.override_height])
        self.end_pos = np.array([0.4, +0.3, self.pick_height])

        #  x-y-z, traj and velocity
        self.x = np.linspace(0.4, 0.4, self.time.shape[0])
        self.y = np.zeros(self.time.shape[0])
        self.z = np.zeros(self.time.shape[0])
        self.trajectory = np.zeros((self.time.shape[0], 3))
        self.velocity = np.zeros((self.time.shape[0], 3))
        self.acceleration = np.zeros((self.time.shape[0], 3))

        # replanning x-y-z, traj and velocity
        self.xr = np.linspace(0.4, 0.4, self.time.shape[0])
        self.yr = np.zeros(self.time.shape[0])
        self.zr = np.zeros(self.time.shape[0])
        self.zr_1 = None
        self.zr_2 = None
        self.trajectory_re = np.zeros((self.time.shape[0], 3))
        self.velocity_re = np.zeros((self.time.shape[0], 3))
        self.acceleration_re = np.zeros((self.time.shape[0], 3))

    def set_time_scope(self, t_max, t_num):
        # reset time
        self.time = np.linspace(0, t_max, t_num)
        self.middle_time_index = int(*np.where(self.time == np.median(self.time)))
        # reset xyz
        # reset trajectory
        self.x = np.linspace(0.4, 0.4, self.time.shape[0])
        self.y = np.zeros(self.time.shape[0])
        self.z = np.zeros(self.time.shape[0])
        self.trajectory = np.zeros((self.time.shape[0], 3))
        self.velocity = np.zeros((self.time.shape[0], 3))
        self.acceleration = np.zeros((self.time.shape[0], 3))

        self.xr = np.linspace(0.4, 0.4, self.time.shape[0])
        self.yr = np.zeros(self.time.shape[0])
        self.zr = np.zeros(self.time.shape[0])
        self.zr_1 = None
        self.zr_2 = None
        self.trajectory_re = np.zeros((self.time.shape[0], 3))
        self.velocity_re = np.zeros((self.time.shape[0], 3))
        self.acceleration_re = np.zeros((self.time.shape[0], 3))

    def set_xyz_scope(self):
        raise NotImplementedError

    def set_replanning(self, replanning_time_index: int, replanning_collision_height: float):
        self.replanning_time_index = int(replanning_time_index * (self.time.shape[0]-1) / 100)
        self.replanning_time = self.time[self.replanning_time_index]
        self.replanning_collision_height = replanning_collision_height
        self.current_collision_height = replanning_collision_height
        self.replanning_override_height = self.pick_height + 2 * self.replanning_collision_height + 0.04

    def get_override_height(self):
        return self.override_height, self.replanning_override_height

    def generate_trajectory(self):
        # x axis
        self.trajectory[:, 0] = self.x
        # y axis
        y_fuc = PolynomialGenerator(4)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = self.y

        # z axis
        z_fuc = PolynomialGenerator(5)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = self.z

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.yr = self.y
        self.trajectory_re[:, 1] = self.yr
        zr_fuc = PolynomialGenerator(5)
        zr_condition = dict()
        # option 1
        zr_condition['pos_condition'] = np.array([[self.replanning_time, z_fuc.compute_output_(self.replanning_time)],
                                                  [np.median(self.time), self.replanning_override_height],
                                                  [self.time[-1], self.end_pos[2]]])
        zr_condition['vel_condition'] = np.array([[self.replanning_time,
                                                   z_fuc.compute_derivative_(self.replanning_time)],
                                                  [np.median(self.time), 0],
                                                  [self.time[-1], 0]])
        zr_fuc.compute_weights(zr_condition)
        # zr_fuc.print_function()
        self.zr = zr_fuc.compute_output(self.time)
        self.trajectory_re[:, 2] = self.zr
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        # print(y_fuc.compute_derivative_(self.time[-1]))
        # print(zr_fuc.compute_derivative_(self.time[-1]))
        return self.trajectory, self.trajectory_re

    def generate_trajectory_option2(self):
        # x axis
        self.trajectory[:, 0] = self.x
        self.velocity[:, 0] = np.zeros(self.x.shape[0])
        # y axis
        y_fuc = PolynomialGenerator(4)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = self.y
        self.velocity[:, 1] = y_fuc.compute_derivative(self.time)

        # z axis
        z_fuc = PolynomialGenerator(5)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = self.z
        self.velocity[:, 2] = z_fuc.compute_derivative(self.time)

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.velocity_re[:, 0] = np.zeros(self.xr.shape[0])
        self.yr = self.y
        self.trajectory_re[:, 1] = self.yr
        self.velocity_re[:, 1] = y_fuc.compute_derivative(self.time)
        zr_fuc = PolynomialGenerator(5)
        zr_condition = dict()
        # option 2
        zr_condition['pos_condition'] = np.array([[self.replanning_time, z_fuc.compute_output_(self.replanning_time)],
                                                  [np.median(self.time), self.replanning_override_height],
                                                  [1-self.replanning_time,
                                                   z_fuc.compute_output_(1-self.replanning_time)]])
        zr_condition['vel_condition'] = np.array([[self.replanning_time,
                                                   z_fuc.compute_derivative_(self.replanning_time)],
                                                  [np.median(self.time), 0],
                                                  [1-self.replanning_time,
                                                   z_fuc.compute_derivative_(1-self.replanning_time)]])
        zr_fuc.compute_weights(zr_condition)
        # zr_fuc.print_function()
        self.zr = zr_fuc.compute_output(self.time)
        self.trajectory_re[:, 2] = self.zr
        self.velocity_re[:, 2] = zr_fuc.compute_derivative(self.time)
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        self.velocity_re[:self.replanning_time_index, :] = self.velocity[:self.replanning_time_index, :]
        # option 2
        self.trajectory_re[101-self.replanning_time_index:, :] = self.trajectory[101-self.replanning_time_index:, :]
        self.velocity_re[101-self.replanning_time_index:, :] = self.velocity[101-self.replanning_time_index:, :]
        # print(y_fuc.compute_derivative_(self.time[-1]))
        # print(zr_fuc.compute_derivative_(self.time[-1]))
        return self.trajectory, self.velocity, self.trajectory_re, self.velocity_re

    def generate_trajectory_option3(self):
        # x axis
        self.trajectory[:, 0] = self.x
        self.velocity[:, 0] = np.zeros(self.x.shape[0])
        # y axis
        y_fuc = PolynomialGenerator(4)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = self.y
        self.velocity[:, 1] = y_fuc.compute_derivative(self.time)

        # z axis
        z_fuc = PolynomialGenerator(5)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = self.z
        self.velocity[:, 2] = z_fuc.compute_derivative(self.time)

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.velocity_re[:, 0] = np.zeros(self.xr.shape[0])
        self.yr = self.y
        self.trajectory_re[:, 1] = self.yr
        self.velocity_re[:, 1] = y_fuc.compute_derivative(self.time)
        z_x = [self.replanning_time, np.median(self.time), self.time[-1]]
        z_y = [z_fuc.compute_output_(self.replanning_time), self.replanning_override_height, self.end_pos[2]]
        z_l = [(1,  z_fuc.compute_derivative_(self.replanning_time)),
               (2, z_fuc.compute_two_order_derivative_(self.replanning_time))]
        z_r = [(1, 0.0),
               (2, z_fuc.compute_two_order_derivative_(self.time[-1]))]
        zr_fuc = make_interp_spline(x=z_x, y=z_y, bc_type=(z_l, z_r), k=5)
        self.zr = zr_fuc(self.time[self.replanning_time_index:])
        self.trajectory_re[self.replanning_time_index:, 2] = self.zr
        # self.velocity_re[:, 2] = zr_fuc.compute_derivative(self.time)
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        self.velocity_re[:self.replanning_time_index, :] = self.velocity[:self.replanning_time_index, :]
        return self.trajectory, self.velocity, self.trajectory_re, self.velocity_re

    def generate_trajectory_option4(self):
        # x axis
        self.trajectory[:, 0] = self.x
        self.velocity[:, 0] = np.zeros(self.x.shape[0])
        # y axis
        y_fuc = PolynomialGenerator(4)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = self.y
        self.velocity[:, 1] = y_fuc.compute_derivative(self.time)

        # z axis
        z_fuc = PolynomialGenerator(5)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = self.z
        self.velocity[:, 2] = z_fuc.compute_derivative(self.time)

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.velocity_re[:, 0] = np.zeros(self.xr.shape[0])
        self.yr = self.y
        self.trajectory_re[:, 1] = self.yr
        self.velocity_re[:, 1] = y_fuc.compute_derivative(self.time)
        # part 1
        z_x_1 = [self.replanning_time, np.median(self.time)]
        z_y_1 = [z_fuc.compute_output_(self.replanning_time), self.replanning_override_height]
        z_l_1 = [(1,  z_fuc.compute_derivative_(self.replanning_time))]
        z_r_1 = [(1, 0.0)]
        zr_fuc_1 = make_interp_spline(x=z_x_1, y=z_y_1, bc_type=(z_l_1, z_r_1), k=3)
        # part 2
        z_x_2 = [np.median(self.time), self.time[-1]]
        z_y_2 = [self.replanning_override_height, self.end_pos[2]]
        z_l_2 = [(1,  0.0)]
        z_r_2 = [(1, 0.0)]
        zr_fuc_2 = make_interp_spline(x=z_x_2, y=z_y_2, bc_type=(z_l_2, z_r_2), k=3)
        self.zr_1 = zr_fuc_1(self.time[self.replanning_time_index: 50])
        self.zr_2 = zr_fuc_2(self.time[50:])
        self.trajectory_re[self.replanning_time_index: 50, 2] = self.zr_1
        self.trajectory_re[50:, 2] = self.zr_2
        # self.velocity_re[:, 2] = zr_fuc.compute_derivative(self.time)
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        self.velocity_re[:self.replanning_time_index, :] = self.velocity[:self.replanning_time_index, :]
        return self.trajectory, self.velocity, self.trajectory_re, self.velocity_re

    def generate_trajectory_option5(self):
        # x axis
        self.trajectory[:, 0] = self.x
        self.velocity[:, 0] = np.zeros(self.x.shape[0])
        # y axis
        y_fuc = PolynomialGenerator(4)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = self.y
        self.velocity[:, 1] = y_fuc.compute_derivative(self.time)

        # z axis
        z_fuc = PolynomialGenerator(5)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = self.z
        self.velocity[:, 2] = z_fuc.compute_derivative(self.time)

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.velocity_re[:, 0] = np.zeros(self.xr.shape[0])
        self.yr = self.y
        self.trajectory_re[:, 1] = self.yr
        self.velocity_re[:, 1] = y_fuc.compute_derivative(self.time)
        # part 1
        zr_fuc_1 = PolynomialGenerator(3)
        zr_1_condition = dict()
        zr_1_condition['pos_condition'] = np.array([[self.replanning_time, z_fuc.compute_output_(self.replanning_time)],
                                                    [np.median(self.time), self.replanning_override_height]])
        zr_1_condition['vel_condition'] = np.array([[self.replanning_time,
                                                     z_fuc.compute_derivative_(self.replanning_time)],
                                                    [np.median(self.time), 0]])
        zr_fuc_1.compute_weights(zr_1_condition)
        # part 2
        zr_fuc_2 = PolynomialGenerator(3)
        zr_2_condition = dict()
        zr_2_condition['pos_condition'] = np.array([[np.median(self.time), self.replanning_override_height],
                                                    [self.time[-1],
                                                     z_fuc.compute_output_(self.time[-1])]])
        zr_2_condition['vel_condition'] = np.array([[np.median(self.time), 0],
                                                    [self.time[-1], 0]])
        zr_fuc_2.compute_weights(zr_2_condition)
        self.zr_1 = zr_fuc_1.compute_output(self.time[self.replanning_time_index: 50])
        self.zr_2 = zr_fuc_2.compute_output(self.time[50:])
        self.trajectory_re[self.replanning_time_index: 50, 2] = self.zr_1
        self.trajectory_re[50:, 2] = self.zr_2
        self.velocity_re[self.replanning_time_index: 50, 2] = \
            zr_fuc_1.compute_derivative(self.time[self.replanning_time_index: 50])
        self.velocity_re[50:, 2] = zr_fuc_2.compute_derivative(self.time[50:])
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        self.velocity_re[:self.replanning_time_index, :] = self.velocity[:self.replanning_time_index, :]
        return self.trajectory, self.velocity, self.trajectory_re, self.velocity_re

    def generate_trajectory_option6(self):
        # x axis
        self.trajectory[:, 0] = self.x
        self.velocity[:, 0] = np.zeros(self.x.shape[0])
        # y axis
        y_fuc = PolynomialGenerator(6)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_condition['acc_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = self.y
        self.velocity[:, 1] = y_fuc.compute_derivative(self.time)

        # z axis
        z_fuc = PolynomialGenerator(7)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_condition['acc_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = self.z
        self.velocity[:, 2] = z_fuc.compute_derivative(self.time)

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.velocity_re[:, 0] = np.zeros(self.xr.shape[0])
        self.yr = self.y
        self.trajectory_re[:, 1] = self.yr
        self.velocity_re[:, 1] = y_fuc.compute_derivative(self.time)
        # part 1
        zr_fuc_1 = PolynomialGenerator(4)
        zr_1_condition = dict()
        zr_1_condition['pos_condition'] = np.array([[self.replanning_time, z_fuc.compute_output_(self.replanning_time)],
                                                    [np.median(self.time), self.replanning_override_height]])
        zr_1_condition['vel_condition'] = np.array([[self.replanning_time,
                                                     z_fuc.compute_derivative_(self.replanning_time)],
                                                    [np.median(self.time), 0]])
        zr_1_condition['acc_condition'] = np.array([[self.replanning_time,
                                                     z_fuc.compute_two_order_derivative_(self.replanning_time)]])
        zr_fuc_1.compute_weights(zr_1_condition)
        # part 2
        zr_fuc_2 = PolynomialGenerator(4)
        zr_2_condition = dict()
        zr_2_condition['pos_condition'] = np.array([[np.median(self.time), self.replanning_override_height],
                                                    [self.time[-1],
                                                     z_fuc.compute_output_(self.time[-1])]])
        zr_2_condition['vel_condition'] = np.array([[np.median(self.time), 0],
                                                    [self.time[-1], 0]])
        zr_2_condition['acc_condition'] = np.array([[self.time[-1], 0]])
        zr_fuc_2.compute_weights(zr_2_condition)
        self.zr_1 = zr_fuc_1.compute_output(self.time[self.replanning_time_index: 50])
        self.zr_2 = zr_fuc_2.compute_output(self.time[50:])
        self.trajectory_re[self.replanning_time_index: 50, 2] = self.zr_1
        self.trajectory_re[50:, 2] = self.zr_2
        self.velocity_re[self.replanning_time_index: 50, 2] = \
            zr_fuc_1.compute_derivative(self.time[self.replanning_time_index: 50])
        self.velocity_re[50:, 2] = zr_fuc_2.compute_derivative(self.time[50:])
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        self.velocity_re[:self.replanning_time_index, :] = self.velocity[:self.replanning_time_index, :]
        return self.trajectory, self.velocity, self.trajectory_re, self.velocity_re

    def generate_trajectory_option7(self):
        # x axis
        self.trajectory[:, 0] = self.x
        self.velocity[:, 0] = np.zeros(self.x.shape[0])
        self.acceleration[:, 0] = np.zeros(self.x.shape[0])
        # y axis
        y_fuc = PolynomialGenerator(6)
        y_condition = dict()
        y_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[1]],
                                                 [np.median(self.time), self.middle_pos[1]],
                                                 [self.time[-1], self.end_pos[1]]])
        y_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_condition['acc_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        y_fuc.compute_weights(y_condition)
        # y_fuc.print_function()
        self.y = y_fuc.compute_output(self.time)
        self.trajectory[:, 1] = y_fuc.compute_output(self.time)
        self.velocity[:, 1] = y_fuc.compute_derivative(self.time)
        self.acceleration[:, 1] = y_fuc.compute_two_order_derivative(self.time)

        # z axis
        z_fuc = PolynomialGenerator(7)
        z_condition = dict()
        z_condition['pos_condition'] = np.array([[self.time[0], self.initial_pos[2]],
                                                 [np.median(self.time), self.override_height],
                                                 [self.time[-1], self.end_pos[2]]])
        z_condition['vel_condition'] = np.array([[self.time[0], 0],
                                                 [np.median(self.time), 0],
                                                 [self.time[-1], 0]])
        z_condition['acc_condition'] = np.array([[self.time[0], 0],
                                                 [self.time[-1], 0]])
        z_fuc.compute_weights(z_condition)
        # z_fuc.print_function()
        self.z = z_fuc.compute_output(self.time)
        self.trajectory[:, 2] = z_fuc.compute_output(self.time)
        self.velocity[:, 2] = z_fuc.compute_derivative(self.time)
        self.acceleration[:, 2] = z_fuc.compute_two_order_derivative(self.time)

        # replanning
        self.trajectory_re[:, 0] = self.xr
        self.velocity_re[:, 0] = np.zeros(self.xr.shape[0])
        self.acceleration_re[:, 0] = np.zeros(self.xr.shape[0])
        self.yr = self.y
        self.trajectory_re[:, 1] = y_fuc.compute_output(self.time)
        self.velocity_re[:, 1] = y_fuc.compute_derivative(self.time)
        self.acceleration_re[:, 1] = y_fuc.compute_two_order_derivative(self.time)
        # part 1
        zr_fuc_1 = PolynomialGenerator(4)
        zr_1_condition = dict()
        zr_1_condition['pos_condition'] = np.array([[self.replanning_time, z_fuc.compute_output_(self.replanning_time)],
                                                    [np.median(self.time), self.replanning_override_height]])
        zr_1_condition['vel_condition'] = np.array([[self.replanning_time,
                                                     z_fuc.compute_derivative_(self.replanning_time)],
                                                    [np.median(self.time), 0]])
        zr_1_condition['acc_condition'] = np.array([[self.replanning_time,
                                                     z_fuc.compute_two_order_derivative_(self.replanning_time)]])
        zr_fuc_1.compute_weights(zr_1_condition)
        # part 2
        zr_fuc_2 = PolynomialGenerator(4)
        zr_2_condition = dict()
        zr_2_condition['pos_condition'] = np.array([[np.median(self.time), self.replanning_override_height],
                                                    [self.time[-1],
                                                     z_fuc.compute_output_(self.time[-1])]])
        zr_2_condition['vel_condition'] = np.array([[np.median(self.time), 0],
                                                    [self.time[-1], 0]])
        zr_2_condition['acc_condition'] = np.array([[self.time[-1], 0]])
        zr_fuc_2.compute_weights(zr_2_condition)
        self.zr_1 = zr_fuc_1.compute_output(self.time[self.replanning_time_index: self.middle_time_index])
        self.zr_2 = zr_fuc_2.compute_output(self.time[self.middle_time_index:])
        self.trajectory_re[self.replanning_time_index: self.middle_time_index, 2] = self.zr_1
        self.trajectory_re[self.middle_time_index:, 2] = self.zr_2
        self.velocity_re[self.replanning_time_index: self.middle_time_index, 2] = \
            zr_fuc_1.compute_derivative(self.time[self.replanning_time_index: self.middle_time_index])
        self.velocity_re[self.middle_time_index:, 2] = zr_fuc_2.compute_derivative(self.time[self.middle_time_index:])
        self.acceleration_re[self.replanning_time_index: self.middle_time_index, 2] = \
            zr_fuc_1.compute_two_order_derivative(self.time[self.replanning_time_index: self.middle_time_index])
        self.acceleration_re[self.middle_time_index:, 2] = \
            zr_fuc_2.compute_two_order_derivative(self.time[self.middle_time_index:])
        self.trajectory_re[:self.replanning_time_index, :] = self.trajectory[:self.replanning_time_index, :]
        self.velocity_re[:self.replanning_time_index, :] = self.velocity[:self.replanning_time_index, :]
        self.acceleration_re[:self.replanning_time_index, :] = self.acceleration[:self.replanning_time_index, :]
        return self.trajectory, self.velocity, self.acceleration, \
            self.trajectory_re, self.velocity_re, self.acceleration_re


class PickAndPlaceDatasetGenerator:
    def __init__(self):
        raise NotImplementedError

def test_polynomial_generator():
    # two dimension
    t = np.linspace(0, 10, 101)

    # x axis
    x_fuc = PolynomialGenerator(4)
    x_condition = dict()
    x_condition['pos_condition'] = np.array([[0, 0], [5, 5], [10, 10]])
    x_condition['vel_condition'] = np.array([[0, 0], [10, 0]])
    x_fuc.compute_weights(x_condition)
    x_fuc.print_function()
    x = x_fuc.compute_output(t)
    dx = x_fuc.compute_derivative(t)

    # y axis
    y_fuc = PolynomialGenerator(5)
    y_condition = dict()
    y_condition['pos_condition'] = np.array([[0, 0], [5, 10], [10, 0]])
    y_condition['vel_condition'] = np.array([[0, 0], [5, 0], [10, 0]])
    y_fuc.compute_weights(y_condition)
    y_fuc.print_function()
    y = y_fuc.compute_output(t)
    dy = y_fuc.compute_derivative(t)

    # re-planing
    xr_fuc = PolynomialGenerator(4)
    xr_condition = dict()
    xr_condition['pos_condition'] = np.array([[2, x_fuc.compute_output_(2)], [5, 5], [10, 10]])
    xr_condition['vel_condition'] = np.array([[2, x_fuc.compute_derivative_(2)], [10, 0]])  # 0.96
    # print(x_fuc.compute_derivative_(2))
    xr_fuc.compute_weights(xr_condition)
    xr_fuc.print_function()
    xr = xr_fuc.compute_output(t)
    dxr = xr_fuc.compute_derivative(t)

    yr_fuc = PolynomialGenerator(5)
    yr_condition = dict()
    yr_condition['pos_condition'] = np.array([[2, y_fuc.compute_output_(2)], [5, 12], [10, 0]])
    yr_condition['vel_condition'] = np.array([[2, y_fuc.compute_derivative_(2)], [5, 0], [10, 0]])  # 3.072
    # print(y_fuc.compute_derivative_(2))
    yr_fuc.compute_weights(yr_condition)
    yr_fuc.print_function()
    yr = yr_fuc.compute_output(t)
    dyr = yr_fuc.compute_derivative(t)

    # plot
    plt.plot(t, x)
    plt.plot(t, xr)
    plt.show()

    plt.plot(t, dx)
    plt.plot(t, dxr)
    plt.show()

    plt.plot(t, y)
    plt.plot(t, yr)
    plt.show()

    plt.plot(t, dy)
    plt.plot(t, dyr)
    plt.show()

    plt.plot(x, y)
    plt.plot(xr[20:], yr[20:])
    plt.show()


def test_trajectory_generator():
    # box height 0.02
    # collision height from 0.02 to 0.10
    box_ctx = 0.02
    # collision_ctx = [0.02, 0.04, 0.06, 0.08, 0.10]
    # replanning_time_ctx = [15, 20, 25, 30, 35]

    collision_ctx = [0.1, 0.15, 0.20, 0.25, 0.30]
    collision_replanning_ctx = [0.1, 0.15, 0.20, 0.25, 0.30]
    replanning_time_ctx = [15, 20, 25, 30, 35]

    t = np.linspace(0, 1, 101)

    # for ctx in collision_ctx:
    #     tg = TrajectoryGenerator(box_ctx, ctx)
    #     for ctx_re in collision_replanning_ctx:
    #         for time_re in replanning_time_ctx:
    #             tg.set_replanning(time_re, ctx_re)
    #             traj, traj_re = tg.generate_trajectory()
    #             # plt.plot(traj[:, 1], traj[:, 2])
    #             plt.plot(traj_re[:, 1], traj_re[:, 2])
    # plt.plot(traj[:, 1], traj[:, 2])
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.68, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.63, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.58, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.53, 100), linestyle='dashed')
    # plt.plot(np.zeros(100), np.linspace(0.4, 0.48, 100), color='b')
    # plt.show()

    for ctx in collision_ctx:
        tg = TrajectoryGenerator(box_ctx, ctx)
        for ctx_re in collision_replanning_ctx:
            for time_re in replanning_time_ctx:
                tg.set_replanning(time_re, ctx_re)
                traj, _, traj_re, vel_re = tg.generate_trajectory_option2()
                # plt.plot(traj[:, 1], traj[:, 2])
                plt.plot(traj_re[:, 1], traj_re[:, 2])
                # plt.plot(t, vel_re[:, 2])
    plt.plot(traj[:, 1], traj[:, 2])
    plt.plot(np.zeros(100), np.linspace(0.4, 0.68, 100), linestyle='dashed')
    plt.plot(np.zeros(100), np.linspace(0.4, 0.63, 100), linestyle='dashed')
    plt.plot(np.zeros(100), np.linspace(0.4, 0.58, 100), linestyle='dashed')
    plt.plot(np.zeros(100), np.linspace(0.4, 0.53, 100), linestyle='dashed')
    plt.plot(np.zeros(100), np.linspace(0.4, 0.48, 100), color='b')
    plt.show()


def test_dataset_generation(to_dataset_name: str):
    # create corresponding dataset dir
    dataset_path = util.get_dataset_dir(to_dataset_name)
    util.remove_file_dir(dataset_path)
    os.makedirs(dataset_path)

    box_ctx = 0.02
    collision_ctx = 3*np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    collision_replanning_ctx = 3*np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    replanning_time_ctx = [15, 20, 25, 30, 35]
    t = np.linspace(0, 1, 101)
    init_c_ctx = np.zeros(101)
    replan_c_ctx = np.zeros(101)
    replan_time_index = np.zeros(101)
    replan_time = np.zeros(101)
    name_num = 0

    # option 1 -- total 125 trajs including 20 repeated trajs
    for coll_ctx in collision_ctx:
        tg = TrajectoryGenerator(box_ctx, coll_ctx)
        for coll_re_ctx in collision_replanning_ctx:
            for time_re in replanning_time_ctx:
                data_dict = dict()
                data_dict['t'] = t
                tg.set_replanning(time_re, coll_re_ctx)
                _, _, traj_re, vel_re = tg.generate_trajectory_option6()
                data_dict['c_pos'] = traj_re
                data_dict['c_vel'] = vel_re
                c_ctx = np.zeros(t.shape[0])
                c_ctx[:time_re] = coll_ctx
                c_ctx[time_re:] = coll_re_ctx
                data_dict['coll_ctx'] = c_ctx
                init_c_ctx[:] = coll_ctx
                data_dict['init_c_ctx'] = init_c_ctx
                replan_c_ctx[:] = coll_re_ctx
                data_dict['replan_c_ctx'] = replan_c_ctx
                replan_time_index[:] = time_re
                data_dict['replan_time_index'] = replan_time_index
                replan_time[:] = t[time_re]
                data_dict['replan_time'] = replan_time
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
                list_ctx_compound_data.append(data_dict['replan_time_index'][0])
                c_pos_ctx = np.zeros((t.shape[0], 2))
                c_pos_ctx[:, 0] = data_dict['c_pos'][time_re, 1]
                c_pos_ctx[:, 1] = data_dict['c_pos'][time_re, 2]
                # list_ctx_compound_data.append(c_pos_ctx[:, 0][0])
                list_ctx_compound_data.append(c_pos_ctx[:, 1][0])
                c_vel_ctx = np.zeros((t.shape[0], 2))
                c_vel_ctx[:, 0] = data_dict['c_vel'][time_re, 1]
                c_vel_ctx[:, 1] = data_dict['c_vel'][time_re, 2]
                # list_ctx_compound_data.append(c_vel_ctx[:, 0][0])
                list_ctx_compound_data.append(c_vel_ctx[:, 1][0])
                list_ctx_compound_data.append(data_dict['replan_c_ctx'][0])
                ctx_compound_data = list(np.stack(list_ctx_compound_data, axis=-1)[None])

                # hw = list(np.stack([height, width], axis=-1))
                # if sparse:
                #     for i in range(1, 5):
                #         y[-i] = np.nan
                #
                # data = pd.DataFrame({"t": t,
                #                      "x": x,
                #                      "y": y,
                #                      "xy": xy})
                #
                # data_static = pd.DataFrame({"height": height,
                #                             "width": width,
                #                             "hw": hw})


                # df['ctx'] = ctx_compound_data
                df_ctx = pandas.DataFrame({'ctx': ctx_compound_data})
                df_ctx.to_csv(path_or_buf=dataset_path+'/static_'+str(name_num),
                              index=False,
                              quoting=csv.QUOTE_ALL)
                df.to_csv(path_or_buf=dataset_path + '/' + str(name_num),
                          index=False,
                          quoting=csv.QUOTE_ALL)
                name_num += 1


if __name__ == '__main__':
    # test
    # test_polynomial_generator()
    # test_trajectory_generator()

    # using option2
    # test_dataset_generation('robot_pick_and_place/exp3/pap_compound')

    # using option5
    # test_dataset_generation('robot_pick_and_place/exp4/pap_compound')

    # using option6
    test_dataset_generation('robot_pick_and_place/exp5/pap_compound')