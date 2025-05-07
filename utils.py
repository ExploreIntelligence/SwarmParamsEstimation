import numpy as np
import pandas as pd
import math

# global parameters
global_initial_follower_position = np.array([[[0.0003, 79.21],[75.33, 24.48],[46.56, -64.08],[-46.56, -64.08],[-75.33, 24.48]]])
global_initial_follower_velocity = np.array([[[10, 0],[10, 0],[10, 0],[10, 0],[10, 0]]])
global_num_virtual_leader = 1
global_num_uav = 5
global_time_length = 100
global_time_step = 0.01
global_noise = 10**(-7)
global_acc = np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]])


def calculate_dist(xi, xj):
    result = math.sqrt((xi[0]-xj[0])**2 + (xi[1]-xj[1])**2)
    return result

def calculate_vi(dist, param_alfa, param_d0, param_d1):
    if dist>0 and dist<param_d1:
        result = param_alfa * (math.log(dist) + param_d0/dist)
    else:
        result = param_alfa * (math.log(param_d1) + param_d0/param_d1)
    return result

def calculate_field(dist, param_alfa, param_d0, param_d1):
    if dist>0 and dist<param_d1:
        result = param_alfa * (1/dist - param_d0/dist**2)
    else:
        result = 0
    return result

def calculate_dissipation_force(v_myself, v_virtual_leader, num_virtual_leader, param_K):
    sum = np.array([0,0])
    for i in range(num_virtual_leader):
        sum = sum + v_virtual_leader[i]
    result = -param_K*(v_myself - (1/num_virtual_leader) * sum)
    return result

def calculate_dynamic_equation(index_uav, position_uav, velocity_uav, num_uav, position_virtual, velocity_virtual, num_virtual_leader, param_alfa, param_d0, param_d1, param_K):
    sum_uav = np.array([0, 0])
    for j in range(num_uav):
        if j == index_uav:
            continue
        dist = calculate_dist(position_uav[index_uav], position_uav[j])
        result = calculate_field(dist, param_alfa, param_d0, param_d1)
        xij = np.array([position_uav[index_uav][0]-position_uav[j][0], position_uav[index_uav][1]-position_uav[j][1]])
        sum_uav = sum_uav + xij * result / dist
    sum_virtual = np.array([0, 0])
    for k in range(num_virtual_leader):
        dist = calculate_dist(position_uav[index_uav], position_virtual[k])
        result = calculate_field(dist, param_alfa, param_d0, param_d1)
        hik = np.array([position_uav[index_uav][0]-position_virtual[k][0], position_uav[index_uav][1]-position_virtual[k][1]])
        sum_virtual = sum_virtual + hik * result / dist  
    result = -sum_uav - sum_virtual + calculate_dissipation_force(velocity_uav[index_uav], velocity_virtual, num_virtual_leader, param_K)
    return result


def get_y_t(pos_y, vel_y):
    '''
    input: pos, vel, shape: [50005][2] (without time step)
    return: y_t, shape: [10001][20]
    '''
    y_dim = 20
    global_time_length = 100
    global_time_step = 0.01
    num_steps = int(global_time_length/global_time_step) + 1
    y_t = np.zeros([num_steps,y_dim])
    for step_idx in range(num_steps):
        # position of agent1
        y_t[step_idx,0:1+1] = pos_y[step_idx*global_num_uav+0,:]
        # velocity of agent1
        y_t[step_idx,2:3+1] = vel_y[step_idx*global_num_uav+0,:]
        # position of agent2
        y_t[step_idx,4:5+1] = pos_y[step_idx*global_num_uav+1,:]
        # velocity of agent2
        y_t[step_idx,6:7+1] = vel_y[step_idx*global_num_uav+1,:]
        # position of agent3
        y_t[step_idx,8:9+1] = pos_y[step_idx*global_num_uav+2,:]
        # velocity of agent3
        y_t[step_idx,10:11+1] = vel_y[step_idx*global_num_uav+2,:]
        # position of agent4
        y_t[step_idx,12:13+1] = pos_y[step_idx*global_num_uav+3,:]
        # velocity of agent4
        y_t[step_idx,14:15+1] = vel_y[step_idx*global_num_uav+3,:]
        # position of agent5
        y_t[step_idx,16:17+1] = pos_y[step_idx*global_num_uav+4,:]
        # velocity of agent5
        y_t[step_idx,18:19+1] = vel_y[step_idx*global_num_uav+4,:]
    return y_t


# generate track based on the given parameters
def generate_track(x0, x1, x2, x3, x4, x5, x6, x7):
    '''
    position = [x0, x1]
    velocity = [x2, x3]
    param_alfa = x4
    param_d0 = x5
    param_d1 = x6
    param_K = x7
    '''
    velocity_uav_all = global_initial_follower_velocity
    position_uav_all = global_initial_follower_position
    position_virtual_all = np.array([[[x0, x1]]])
    velocity_virtual_all = np.array([[[x2, x3]]])
    param_alfa = x4
    param_d0 = x5
    param_d1 = x6
    param_K = x7
    acc = global_acc


    for t in range(round(global_time_length/global_time_step)):
        for i in range(global_num_uav):
            velocity_uav_alone_temp = np.array([velocity_uav_all[t][i]]) + acc[i] * global_time_step
            position_uav_alone_temp = velocity_uav_alone_temp * global_time_step + np.array([position_uav_all[t][i]])
            if i==0:
                velocity_uav_alone = velocity_uav_alone_temp
                position_uav_alone = position_uav_alone_temp
            else:
                velocity_uav_alone = np.append(velocity_uav_alone,velocity_uav_alone_temp,0)
                position_uav_alone = np.append(position_uav_alone,position_uav_alone_temp,0)        
        velocity_uav_all = np.append(velocity_uav_all, np.array([velocity_uav_alone]), 0)
        position_uav_all = np.append(position_uav_all, np.array([position_uav_alone]), 0)
        for k in range(global_num_virtual_leader):
            velocity_virtual_alone_temp = np.array([velocity_virtual_all[t][k]])
            position_virtual_alone_temp = np.array([position_virtual_all[t][k]]) + velocity_virtual_alone_temp * global_time_step
            if k==0:
                velocity_virtual_alone = velocity_virtual_alone_temp
                position_virtual_alone = position_virtual_alone_temp
            else:
                velocity_virtual_alone = np.append(velocity_virtual_alone,velocity_virtual_alone_temp,0)
                position_virtual_alone = np.append(position_virtual_alone,position_virtual_alone_temp,0)
        
        velocity_virtual_all = np.append(velocity_virtual_all, np.array([velocity_virtual_alone]), 0)
        position_virtual_all = np.append(position_virtual_all, np.array([position_virtual_alone]), 0)  

        for i in range(global_num_uav):
            acc[i] = calculate_dynamic_equation(i, position_uav_all[t+1], velocity_uav_all[t+1], global_num_uav, position_virtual_all[t+1], velocity_virtual_all[t+1], global_num_virtual_leader, param_alfa, param_d0, param_d1, param_K)
            
    trajectory_position = np.reshape(position_uav_all,(-1,2))
    trajectory_velocity = np.reshape(velocity_uav_all,(-1,2))
    y_t = get_y_t(trajectory_position, trajectory_velocity)
    return y_t


def get_real_track():
    # read csv file
    pos = pd.read_csv('output/uav_track_position_0.csv').values
    vel = pd.read_csv('output/uav_track_velocity_0.csv').values
    data = get_y_t(pos[:, 1:], vel[:, 1:])
    return data


def cal_mse_loss(y_t, y_hat_t):
    mse_loss = y_t - y_hat_t
    W_y = np.array([1,1,10,10]*5)
    mse_loss = mse_loss * W_y
    mse_loss = np.power(mse_loss,2)
    mse_loss = np.sum(mse_loss)
    mse_loss = mse_loss/len(y_t)
    mse_loss = math.log(1+mse_loss)
    return mse_loss


if __name__ == "__main__":
    print(cal_mse_loss(get_real_track(), generate_track(0, 0, 10, 0, 150, 100, 200, 1)))   # to check the real_track data
    # xbest=array([3.71571796e-03, 1.34371097e-03, 9.99996827e+00, -4.53141245e-06, 1.48201987e+02, 1.00000615e+02, 2.00491643e+02, 9.98688653e-01])