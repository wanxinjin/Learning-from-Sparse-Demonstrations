#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import math
import json
import CPDP
import JinEnv
from casadi import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from QuadPara import QuadPara
from QuadStates import QuadStates
from DemoSparse import DemoSparse
from ObsInfo import ObsInfo
from generate_random_obs import generate_random_obs


class QuadAlgorithm(object):
    learning_rate: float # the learning rate
    iter_num: int # the maximum iteration number
    n_grid: int # the number of grid for nonlinear programming
    QuadPara: QuadPara # the dataclass QuadPara including the quadrotor parameters


    def __init__(self, config_data, QuadParaInput: QuadPara, learning_rate: float, iter_num: int, n_grid: int):
        """
        constructor

        config_data:
            config_file_name = "config.json"
            json_file = open(config_file_name)
            config_data = json.load(json_file)
        """

        self.QuadPara = QuadParaInput
        self.learning_rate = learning_rate
        self.iter_num = iter_num
        self.n_grid = n_grid

        # the lab space limit [meter] in x-axis [x_min, x_max]
        self.space_limit_x = config_data["LAB_SPACE_LIMIT"]["LIMIT_X"]
        # the lab space limit [meter] in y-axis [y_min, y_max]
        self.space_limit_y = config_data["LAB_SPACE_LIMIT"]["LIMIT_Y"]
        # the lab space limit [meter] in z-axis [z_min, z_max]
        self.space_limit_z = config_data["LAB_SPACE_LIMIT"]["LIMIT_Z"]
        # the average speed for the quadrotor [m/s]
        self.quad_average_speed = float(config_data["QUAD_AVERAGE_SPEED"])


    def settings(self, QuadDesiredStates: QuadStates):
        """
        Do the settings and defined the goal states.
        Rerun this function everytime the initial condition or goal states change.
        """

        # load environment
        self.env = JinEnv.Quadrotor()
        self.env.initDyn(Jx=self.QuadPara.inertial_x, Jy=self.QuadPara.inertial_y, Jz=self.QuadPara.inertial_z, \
            mass=self.QuadPara.mass, l=self.QuadPara.l, c=self.QuadPara.c)
        # set the desired goal states
        self.env.initCost_Polynomial(QuadDesiredStates, w_thrust=0.1)

        # create UAV optimal control object with time-warping function
        self.oc = CPDP.COCSys()
        beta = SX.sym('beta')
        dyn = beta * self.env.f
        self.oc.setAuxvarVariable(vertcat(beta, self.env.cost_auxvar))
        self.oc.setStateVariable(self.env.X)
        self.oc.setControlVariable(self.env.U)
        self.oc.setDyn(dyn)
        path_cost = beta * self.env.path_cost
        self.oc.setPathCost(path_cost)
        self.oc.setFinalCost(self.env.final_cost)
        self.oc.setIntegrator(self.n_grid)

        # define the loss function and interface function
        self.interface_pos_fn = Function('interface', [self.oc.state], [self.oc.state[0:3]])
        self.interface_ori_fn = Function('interface', [self.oc.state], [self.oc.state[6:10]])
        self.diff_interface_pos_fn = Function('diff_interface', [self.oc.state], [jacobian(self.oc.state[0:3], self.oc.state)])
        self.diff_interface_ori_fn = Function('diff_interface', [self.oc.state], [jacobian(self.oc.state[6:10], self.oc.state)])


    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, SparseInput: DemoSparse, ObsList: list, print_flag: bool, save_flag: bool):
        """
        Run the algorithm
        """

        print("Algorithm is running now.")
        # set the obstacles for plotting
        self.ObsList = ObsList

        # set the goal states
        self.settings(QuadDesiredStates)

        # set initial condition
        ini_r_I = QuadInitialCondition.position
        ini_v_I = QuadInitialCondition.velocity
        ini_q = QuadInitialCondition.attitude_quaternion
        ini_w = QuadInitialCondition.angular_velocity
        ini_state = ini_r_I + ini_v_I + ini_q + ini_w

        # create sparse waypionts and time horizon
        T = SparseInput.time_horizon
        taus = np.array(SparseInput.time_list)
        waypoints = np.array(SparseInput.waypoints)

        # start the learning process
        loss_trace, parameter_trace = [], []
        current_parameter = np.array([1, 0.1, 0.1, 0.1, 0.1, 0.1, -1])
        parameter_trace += [current_parameter.tolist()]

        diff_loss_norm = 100
        epsilon = 1e-2
        for j in range(self.iter_num):
            if diff_loss_norm >= epsilon:
                time_grid, opt_sol = self.oc.cocSolver(ini_state, T, current_parameter)
                auxsys_sol = self.oc.auxSysSolver(time_grid, opt_sol, current_parameter)
                loss, diff_loss = self.getloss_pos_corrections(taus, waypoints, opt_sol, auxsys_sol)
                current_parameter -= self.learning_rate * diff_loss

                # do the projection step
                current_parameter[0] = fmax(current_parameter[0], 0.00000001)
                loss_trace += [loss]
                parameter_trace += [current_parameter.tolist()]
                
                diff_loss_norm = np.linalg.norm(diff_loss)
                if print_flag:
                    print('iter:', j, ', loss:', loss_trace[-1].tolist(), ', loss gradient norm:',diff_loss_norm)
            else:
                print("The magnitude of gradient of loss is less than epsilon, stop the iteration.")
                break

        # Below is to obtain the final uav trajectory based on the learned objective function (under un-warping settings)
        
        # note this is the uav actual horizon after warping (T is before warping)
        # floor the horizon with 2 decimal
        horizon = math.floor(current_parameter[0]*T*100) / 100.0
        # the learned cost function, but set the time-warping function as unit (un-warping)
        current_parameter[0] = 1
        _, opt_sol = self.oc.cocSolver(ini_state, horizon, current_parameter)
        
        # generate the time inquiry grid with N is the point number
        # time_steps = np.linspace(0, horizon, num=100)
        # time_steps = np.linspace(0, math.floor(horizon*100)/100.0, num=int(math.floor(horizon*100)+1))
        time_steps = np.linspace(0, horizon, num=int(horizon/0.01 +1))

        opt_traj = opt_sol(time_steps)
        
        # state trajectory ----- N*[r,v,q,w]
        opt_state_traj = opt_traj[:, :self.oc.n_state]
        # control trajectory ---- N*[t1,t2,t3,t4]
        opt_control_traj = opt_traj[:, self.oc.n_state : self.oc.n_state + self.oc.n_control]

        if save_flag:
            # save the results
            save_data = {'parameter_trace': parameter_trace,
                        'loss_trace': loss_trace,
                        'learning_rate': self.learning_rate,
                        'waypoints': waypoints,
                        'time_grid': taus,
                        'time_steps': time_steps,
                        'opt_state_traj': opt_state_traj,
                        'opt_control_traj': opt_control_traj,
                        'horizon': horizon,
                        'T': T}

            time_prefix = time.strftime("%Y%m%d%H%M%S")

            # save the results as mat files
            name_prefix_mat = os.getcwd() + '/data/uav_results_random_' + time_prefix
            sio.savemat(name_prefix_mat + '.mat', {'results': save_data})

            # save the trajectory as csv files
            name_prefix_csv = os.getcwd() + '/trajectories/' + time_prefix + '.csv'
            # convert 2d list to 2d numpy array, and slice the first 6 rows
            # num_points by 13 states, but I need states by num_points
            opt_state_traj_numpy = np.array(opt_state_traj)
            csv_np_array = np.concatenate(( np.array([time_steps]), np.transpose(opt_state_traj_numpy[:,0:6]) ) , axis=0)
            np.savetxt(name_prefix_csv, csv_np_array, delimiter=",")

            # plot trajectory in 3D space
            self.plot_opt_trajectory(opt_state_traj_numpy, QuadInitialCondition, QuadDesiredStates, SparseInput)

            #self.env.play_animation(self.QuadPara.l, opt_state_traj, name_prefix, save_option=True)


    def getloss_pos_corrections(self, time_grid, target_waypoints, opt_sol, auxsys_sol):
        loss = 0
        diff_loss = np.zeros(self.oc.n_auxvar)

        for k, t in enumerate(time_grid):
            # solve loss
            target_waypoint = target_waypoints[k, :]
            target_position = target_waypoint[0:3]
            current_position = self.interface_pos_fn(opt_sol(t)[0:self.oc.n_state]).full().flatten()

            loss += np.linalg.norm(target_position - current_position) ** 2
            # solve gradient by chain rule
            dl_dpos = current_position - target_position
            dpos_dx = self.diff_interface_pos_fn(opt_sol(t)[0:self.oc.n_state]).full()
            dxpos_dp = auxsys_sol(t)[0:self.oc.n_state * self.oc.n_auxvar].reshape((self.oc.n_state, self.oc.n_auxvar))

            dl_dp = np.matmul(np.matmul(dl_dpos, dpos_dx), dxpos_dp)
            diff_loss += dl_dp

        return loss, diff_loss

    
    def getloss_corrections(self, time_grid, target_waypoints, opt_sol, auxsys_sol):
        loss = 0
        diff_loss = np.zeros(self.oc.n_auxvar)

        for k, t in enumerate(time_grid):
            # solve loss
            target_waypoint = target_waypoints[k, :]
            target_position = target_waypoint[0:3]
            target_orientation = target_waypoint[3:]
            current_position = self.interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()
            current_orientation = self.interface_ori_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

            loss += np.linalg.norm(target_position - current_position) ** 2 + \
                np.linalg.norm(target_orientation - current_orientation) ** 2
            # solve gradient by chain rule
            dl_dpos = current_position - target_position
            dpos_dx = self.diff_interface_pos_fn(opt_sol(t)[0:self.oc.n_state]).full()
            dxpos_dp = auxsys_sol(t)[0:self.oc.n_state * self.oc.n_auxvar].reshape((self.oc.n_state, self.oc.n_auxvar))

            dl_dori = current_orientation - target_orientation
            dori_dx = self.diff_interface_ori_fn(opt_sol(t)[0:self.oc.n_state]).full()
            dxori_dp = auxsys_sol(t)[0:self.oc.n_state * self.oc.n_auxvar].reshape((self.oc.n_state, self.oc.n_auxvar))

            dl_dp = np.matmul(np.matmul(dl_dpos, dpos_dx), dxpos_dp) + \
                np.matmul(np.matmul(dl_dori, dori_dx),dxori_dp)
            diff_loss += dl_dp

        return loss, diff_loss


    def plot_opt_trajectory(self, opt_state_traj_numpy, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, SparseInput: DemoSparse):
        """
        Plot trajectory and waypoints in 3D space with obstacles.

        opt_state_traj_numpy is a 2D numpy array, each row is all the states at the same time-stamp.
        """

        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        # plot waypoints
        self.ax_3d.plot3D(opt_state_traj_numpy[:,0].tolist(), opt_state_traj_numpy[:,1].tolist(), opt_state_traj_numpy[:,2].tolist(), 'blue', label='optimal trajectory')
        for i in range(0, len(SparseInput.waypoints)):
            self.ax_3d.scatter(SparseInput.waypoints[i][0], SparseInput.waypoints[i][1], SparseInput.waypoints[i][2], c='C0')

        # plot start and goal
        self.ax_3d.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[1], QuadInitialCondition.position[2], label='start', color='green')
        self.ax_3d.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[1], QuadDesiredStates.position[2], label='goal', color='violet')

        # plot obstacles
        self.plot_linear_cube()
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        self.ax_3d.add_artist(plt.legend(handles=[red_patch]))

        self.ax_3d.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        self.ax_3d.set_ylim([self.space_limit_y[0]-1.2, self.space_limit_y[1]+1.2])
        self.ax_3d.set_zlim([self.space_limit_z[0]-0.2, self.space_limit_z[1]+1.5])
        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        plt.legend(loc="upper left")
        plt.title('Trajectory in 3D space.', fontweight ='bold')
        plt.show()

    
    def plot_linear_cube(self, color='red'):
        """
        Plot obstacles in 3D space.
        """

        # plot obstacles
        num_obs = len(self.ObsList)
        if num_obs > 0.5:
            for i in range(0, num_obs):
                x = self.ObsList[i].center[0] - 0.5 * self.ObsList[i].length
                y = self.ObsList[i].center[1] - 0.5 * self.ObsList[i].width
                z = self.ObsList[i].center[2] - 0.5 * self.ObsList[i].height

                dx = self.ObsList[i].length
                dy = self.ObsList[i].width
                dz = self.ObsList[i].height

                xx = [x, x, x+dx, x+dx, x]
                yy = [y, y+dy, y+dy, y, y]
                kwargs = {'alpha': 1, 'color': color}
                self.ax_3d.plot3D(xx, yy, [z]*5, **kwargs)
                self.ax_3d.plot3D(xx, yy, [z+dz]*5, **kwargs)
                self.ax_3d.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
                self.ax_3d.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
                self.ax_3d.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
                self.ax_3d.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)