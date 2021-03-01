#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import time
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
    iter_num: int # the maximum iteration number
    n_grid: int # the number of grid for nonlinear programming
    QuadPara: QuadPara # the dataclass QuadPara including the quadrotor parameters
    ini_state: list # initial states for in a 1D list, [posi, velo, quaternion, angular_velo]
    time_horizon: float # total time [sec] for sparse demonstration (waypoints)

    learning_rate: float # the learning rate
    optimization_method_str: str # a string of optimization method for learning process
    mu_momentum: float # momentum parameter, usually around 0.9, 0 < mu_momentum < 1


    def __init__(self, config_data, QuadParaInput: QuadPara, n_grid: int):
        """
        constructor

        config_data:
            config_file_name = "config.json"
            json_file = open(config_file_name)
            config_data = json.load(json_file)
        """

        self.QuadPara = QuadParaInput
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


    def load_optimization_method(self, para_input: dict):
        """
        Load the optimization method. Now support Vanilla gradient descent or Nesterov Momentum.

        Input:
            para_input: a dictionary which includes the parameters.
        
        Example:
            # This is for Vanilla gradient descent
            para_input = {"learning_rate": 0.01, "iter_num": 1000, "method": "Vanilla"}
            
            # This is for Nesterov Momentum
            para_input = {"learning_rate": 0.01, "iter_num": 1000, "method": "Nesterov", "mu": 0.9}
        """

        # learning rate
        self.learning_rate = para_input["learning_rate"]
        # maximum iteration number
        self.iter_num = para_input["iter_num"]
        # the optimization method
        if (para_input["method"] == "Vanilla"):
            self.optimization_method_str = para_input["method"]
        elif (para_input["method"] == "Nesterov"):
            self.optimization_method_str = para_input["method"]
            self.mu_momentum = para_input["mu"]
        else:
            raise Exception("Wrong optimization method type!")


    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, SparseInput: DemoSparse, ObsList: list, print_flag: bool, save_flag: bool):
        """
        Run the algorithm
        """

        t0 = time.time()
        print("Algorithm is running now.")
        # set the obstacles for plotting
        self.ObsList = ObsList
        # set the goal states
        self.settings(QuadDesiredStates)
        # set initial condition
        self.ini_state = QuadInitialCondition.position + QuadInitialCondition.velocity + \
            QuadInitialCondition.attitude_quaternion + QuadInitialCondition.angular_velocity

        # create sparse waypionts and time horizon
        self.time_horizon = SparseInput.time_horizon
        # time_list_sparse is a numpy 1D array, timestamps [sec] for sparse demonstration (waypoints), not including the start and goal
        self.time_list_sparse = np.array(SparseInput.time_list)
        # waypoints is a numpy 2D array, each row is a waypoint in R^3, i.e. [px, py, pz]
        self.waypoints = np.array(SparseInput.waypoints)

        # test why trajectory doesn't go to the goal, maybe waypoints and time_list should include goal
        #############################################
        # self.waypoints = np.array(SparseInput.waypoints+[QuadDesiredStates.position])


        # for debugging
        self.time_horizon = 1.0
        self.time_list_sparse = np.array(SparseInput.time_list) / SparseInput.time_horizon
        print("T")
        print(self.time_horizon)
        print("taus")
        print(self.time_list_sparse)
        print("waypoints")
        print(self.waypoints)


        # for comparison only
        loss_trace_vanilla = []
        self.comp_flag = False
        self.true_loss_flag = False


        # start the learning process
        # initialize parameter vector and momentum velocity vector
        loss_trace, parameter_trace = [], []
        self.current_parameter = np.array([1, 0.1, 0.1, 0.1, 0.1, 0.1, -1])
        # momentum velocity vector, 1D numpy array
        self.velocity_momentum = np.array([0] * self.current_parameter.shape[0])
        parameter_trace += [self.current_parameter.tolist()]

        loss = 100
        diff_loss_norm = 100
        for j in range(self.iter_num):
            if (loss > 1.0) and (diff_loss_norm > 0.04):

                # update parameter and compute loss, optimization_method_str: "Vanilla" or "Nesterov"
                loss, diff_loss = self.gradient_descent_choose(self.optimization_method_str)

                if self.comp_flag:
                    # for comparison only
                    loss_vanilla, diff_loss_vanilla = self.gradient_descent_choose("Vanilla")
                    loss_trace_vanilla += [loss_vanilla]


                # do the projection step
                self.current_parameter[0] = fmax(self.current_parameter[0], 1e-8)
                loss_trace += [loss]
                parameter_trace += [self.current_parameter.tolist()]
                
                diff_loss_norm = np.linalg.norm(diff_loss)
                if print_flag:
                    print('iter:', j, ', loss:', loss_trace[-1].tolist(), ', loss gradient norm:', diff_loss_norm)
            else:
                print("The loss is less than threshold, stop the iteration.")
                break


        fig_comp = plt.figure()
        ax_comp = fig_comp.add_subplot(111)
        # plot loss
        iter_list = range(0, len(loss_trace))
        ax_comp.plot(iter_list, loss_trace, linewidth=2, color="red", label="Nesterov")
        # for comparison only
        if self.comp_flag:
            ax_comp.plot(iter_list, loss_trace_vanilla, linewidth=2, color="blue", label="Vanilla")
        ax_comp.set_xlabel("Iterations")
        ax_comp.set_ylabel("loss")
        plt.legend(loc="upper right")
        plt.title('Loss (Vanilla vs Nesterov).', fontweight ='bold')
        plt.draw()


        # Below is to obtain the final uav trajectory based on the learned objective function (under un-warping settings)
        
        # note this is the uav actual horizon after warping (T is before warping)
        # floor the horizon with 2 decimal


        # horizon = math.floor(current_parameter[0]*T*100) / 100.0
        # debugging
        horizon = self.time_horizon


        # the learned cost function, but set the time-warping function as unit (un-warping)

        print("beta")
        print(self.current_parameter[0])
        print("horizon")
        print(horizon)


        # current_parameter[0] = 1


        _, opt_sol = self.oc.cocSolver(self.ini_state, horizon, self.current_parameter)
        
        # generate the time inquiry grid with N is the point number
        time_steps = np.linspace(0, horizon, num=100+1)
        # time_steps = np.linspace(0, horizon, num=int(horizon/0.01 +1))

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
                        'waypoints': self.waypoints,
                        'time_grid': self.time_list_sparse,
                        'time_steps': time_steps,
                        'opt_state_traj': opt_state_traj,
                        'opt_control_traj': opt_control_traj,
                        'horizon': horizon,
                        'T': self.time_horizon}

            time_prefix = time.strftime("%Y%m%d%H%M%S")

            # save the results as mat files
            name_prefix_mat = os.getcwd() + '/data/uav_results_random_' + time_prefix
            sio.savemat(name_prefix_mat + '.mat', {'results': save_data})

            # save the trajectory as csv files
            name_prefix_csv = os.getcwd() + '/trajectories/' + time_prefix + '.csv'
            # convert 2d list to 2d numpy array, and slice the first 6 rows
            # num_points by 13 states, but I need states by num_points
            opt_state_traj_numpy = np.array(opt_state_traj)

            posi_velo_traj_numpy = np.transpose(opt_state_traj_numpy[:,0:6])
            csv_np_array = np.concatenate(( np.array([time_steps]), posi_velo_traj_numpy ) , axis=0)
            np.savetxt(name_prefix_csv, csv_np_array, delimiter=",")

            print("time_steps")
            print(np.array([time_steps]))

            t1 = time.time()
            print("Time used [min]: ", (t1-t0)/60)

            # plot trajectory in 3D space
            self.plot_opt_trajectory(posi_velo_traj_numpy, QuadInitialCondition, QuadDesiredStates, SparseInput)

            # play animation
            print("Playing animation")
            name_prefix_animation = os.getcwd() + '/trajectories/animation_' + time_prefix
            space_limits = [self.space_limit_x, self.space_limit_y, self.space_limit_z]
            self.env.play_animation(self.QuadPara.l, opt_state_traj_numpy, name_prefix_animation, space_limits, save_option=True)


    def gradient_descent_choose(self, method_string: str):
        """
        Choose which gradient descent method to use.

        Input:
            method_string: "Vanilla" or "Nesterov"
        """

        if method_string == "Vanilla":

            # vanilla gradient descent method
            time_grid, opt_sol = self.oc.cocSolver(self.ini_state, self.time_horizon, self.current_parameter)
            auxsys_sol = self.oc.auxSysSolver(time_grid, opt_sol, self.current_parameter)
            loss, diff_loss = self.getloss_pos_corrections(self.time_list_sparse, self.waypoints, opt_sol, auxsys_sol)
            self.current_parameter -= self.learning_rate * diff_loss

        elif method_string == "Nesterov":

            # compute the lookahead parameter
            parameter_momentum = self.current_parameter + self.mu_momentum * self.velocity_momentum
            # update velocity_momentum
            time_grid, opt_sol = self.oc.cocSolver(self.ini_state, self.time_horizon, parameter_momentum)
            auxsys_sol = self.oc.auxSysSolver(time_grid, opt_sol, parameter_momentum)
            # only need the gradient
            loss, diff_loss = self.getloss_pos_corrections(self.time_list_sparse, self.waypoints, opt_sol, auxsys_sol)
            self.velocity_momentum = self.mu_momentum * self.velocity_momentum - self.learning_rate * diff_loss
            # update the parameter
            self.current_parameter = self.current_parameter + self.velocity_momentum

            if self.true_loss_flag:
                # t0 = time.time()
                # compute loss and gradient for new parameter
                time_grid, opt_sol = self.oc.cocSolver(self.ini_state, self.time_horizon, self.current_parameter)
                auxsys_sol = self.oc.auxSysSolver(time_grid, opt_sol, self.current_parameter)
                loss, diff_loss = self.getloss_pos_corrections(self.time_list_sparse, self.waypoints, opt_sol, auxsys_sol)
                # t1 = time.time()
                # print("Check time [sec]: ", t1-t0)

        else:
            raise Exception("Wrong type of gradient descent method, only support Vanilla or Nesterov!")

        return loss, diff_loss


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


    def plot_opt_trajectory(self, posi_velo_traj_numpy, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, SparseInput: DemoSparse):
        """
        Plot trajectory and waypoints in 3D space with obstacles.

        posi_velo_traj_numpy is a 2D numpy array, num_states by time_steps. Each column is all states at time t.
        """

        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        # plot waypoints
        self.ax_3d.plot3D(posi_velo_traj_numpy[0,:].tolist(), posi_velo_traj_numpy[1,:].tolist(), posi_velo_traj_numpy[2,:].tolist(), 'blue', label='optimal trajectory')
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

        self.set_axes_equal_all()
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

    
    def set_axes_equal_all(self):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        Reference: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        '''

        x_limits = [self.space_limit_x[0], self.space_limit_x[1]]
        y_limits = [self.space_limit_y[0], self.space_limit_y[1]]
        z_limits = [self.space_limit_z[0], self.space_limit_z[1]]

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax_3d.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax_3d.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax_3d.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
