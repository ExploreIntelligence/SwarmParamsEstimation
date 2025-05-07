import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from numpy.distutils.system_info import accelerate_info

global_initial_leader_position = np.array([[[0, 0]]])
global_initial_leader_velocity = np.array([[[10, 0]]])
global_initial_follower_position = np.array(
    [[[0.0003, 79.21], [75.33, 24.48], [46.56, -64.08], [-46.56, -64.08], [-75.33, 24.48]]])
global_initial_follower_velocity = np.array([[[10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]])
global_num_virtual_leader = 1
global_num_uav = 5
global_time_length = 100
global_time_step = 0.01
global_noise = 10 ** (-7)
global_d0 = 100
global_d1 = 200
global_alfa = 150
global_K = 1
global_acc = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # 五个无人机，速度都是二维


def calculate_dist(xi, xj):
    result = math.sqrt((xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2)
    return result


def calculate_vi(dist, param_alfa, param_d0, param_d1):
    if dist > 0 and dist < param_d1:
        result = param_alfa * (math.log(dist) + param_d0 / dist)
    else:
        result = param_alfa * (math.log(param_d1) + param_d0 / param_d1)
    return result


def calculate_field(dist, param_alfa, param_d0, param_d1):
    if dist > 0 and dist < param_d1:
        result = param_alfa * (1 / dist - param_d0 / dist ** 2)
    else:
        result = 0
    return result


def calculate_dissipation_force(v_myself, v_virtual_leader, num_virtual_leader, param_K):
    sum = np.array([0, 0])  # 二维
    for i in range(num_virtual_leader):
        sum = sum + v_virtual_leader[i]
    result = -param_K * (v_myself - (1 / num_virtual_leader) * sum)
    return result


def calculate_dynamic_equation(index_uav, position_uav, velocity_uav, num_uav, position_virtual, velocity_virtual,
                               num_virtual_leader, param_alfa, param_d0, param_d1, param_K, t):

    sum_uav = np.array([0, 0])
    for j in range(num_uav):
        if j == index_uav:
            continue
        dist = calculate_dist(position_uav[index_uav], position_uav[j])

        if t == 9000:
            print("at ", t, "point: ")
            print(j, "--", index_uav, "dist: ", dist)

        result1 = calculate_field(dist, param_alfa, param_d0, param_d1)
        xij = np.array(
            [position_uav[index_uav][0] - position_uav[j][0], position_uav[index_uav][1] - position_uav[j][1]])  # 正负号修改
        sum_uav = sum_uav + xij * result1 / dist

        if t == 9000:
            print(result1)
            print(xij)

    sum_virtual = np.array([0, 0])
    for k in range(num_virtual_leader):
        dist = calculate_dist(position_uav[index_uav], position_virtual[k])
        result2 = calculate_field(dist, param_alfa, param_d0, param_d1)
        hik = np.array([position_uav[index_uav][0] - position_virtual[k][0],
                        position_uav[index_uav][1] - position_virtual[k][1]])  # 正负号修改
        sum_virtual = sum_virtual + hik * result2 / dist
    result = -sum_uav - sum_virtual + calculate_dissipation_force(velocity_uav[index_uav], velocity_virtual,
                                                                  num_virtual_leader, param_K)

    if t == 9000:
        print("node", index_uav, ":")
        print("acc_1: ", result1, "acc_2: ", result2, "acc_3: ",
              calculate_dissipation_force(velocity_uav[index_uav], velocity_virtual, num_virtual_leader, param_K))
        print("acc_1: ", -sum_uav, "acc_2: ", -sum_virtual, "acc_3: ",
              calculate_dissipation_force(velocity_uav[index_uav], velocity_virtual, num_virtual_leader, param_K))
        print("\n")
    return result


def main():
    for m in range(1):
        if m == 0:
            velocity_uav_all = global_initial_follower_velocity
            position_uav_all = global_initial_follower_position
            velocity_virtual_all = global_initial_leader_velocity
            position_virtual_all = global_initial_leader_position
            param_alfa = global_alfa
            param_d0 = global_d0
            param_d1 = global_d1
            param_K = global_K

        acc = global_acc
        for t in range(round(global_time_length / global_time_step)):
            for i in range(global_num_uav):
                velocity_uav_alone_temp = np.array([velocity_uav_all[t][i]]) + acc[i] * global_time_step
                position_uav_alone_temp = velocity_uav_alone_temp * global_time_step + np.array(
                    [position_uav_all[t][i]])
                if i == 0:
                    velocity_uav_alone = velocity_uav_alone_temp
                    position_uav_alone = position_uav_alone_temp
                else:
                    velocity_uav_alone = np.append(velocity_uav_alone, velocity_uav_alone_temp, 0)
                    position_uav_alone = np.append(position_uav_alone, position_uav_alone_temp, 0)
            velocity_uav_all = np.append(velocity_uav_all, np.array([velocity_uav_alone]), 0)
            position_uav_all = np.append(position_uav_all, np.array([position_uav_alone]), 0)
            for k in range(global_num_virtual_leader):
                velocity_virtual_alone_temp = np.array([velocity_virtual_all[t][k]])
                position_virtual_alone_temp = np.array(
                    [position_virtual_all[t][k]]) + velocity_virtual_alone_temp * global_time_step
                if k == 0:
                    velocity_virtual_alone = velocity_virtual_alone_temp
                    position_virtual_alone = position_virtual_alone_temp
                else:
                    velocity_virtual_alone = np.append(velocity_virtual_alone, velocity_virtual_alone_temp, 0)
                    position_virtual_alone = np.append(position_virtual_alone, position_virtual_alone_temp, 0)
            velocity_virtual_all = np.append(velocity_virtual_all, np.array([velocity_virtual_alone]), 0)
            position_virtual_all = np.append(position_virtual_all, np.array([position_virtual_alone]), 0)
            for i in range(global_num_uav):
                acc[i] = calculate_dynamic_equation(i, position_uav_all[t + 1], velocity_uav_all[t + 1], global_num_uav,
                                                    position_virtual_all[t + 1], velocity_virtual_all[t + 1],
                                                    global_num_virtual_leader, param_alfa, param_d0, param_d1, param_K,
                                                    t)


if __name__ == "__main__":
    main()
    print("End\n")