#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import time
import json
import numpy as np
import transforms3d
from dataclasses import dataclass, field
from QuadPara import QuadPara
from QuadStates import QuadStates
from DemoSparse import DemoSparse
from QuadAlgorithm import QuadAlgorithm
from InputWaypoints import InputWaypoints
from ObsInfo import ObsInfo
from generate_random_obs import generate_random_obs


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # generate random obstacles
    num_obs = 20 # number of obstacles
    size_list=[0.2, 0.3, 0.4] # size lenth, width, height in x,y,z axis
    ObsList = generate_random_obs(num_obs, size_list, config_data)

    # define the quadrotor dynamics parameters
    QuadParaInput = QuadPara(inertial_list=[1.0, 1.0, 1.0], mass=1.0, l=1.0, c=0.02)

    # number of grids for nonlinear programming solver
    n_grid = 25

    # define the initial condition
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    QuadInitialCondition = QuadStates()
    QuadInitialCondition.position = [-2.0, -1.0, 0.6]
    QuadInitialCondition.velocity = [0, 0, 0]
    QuadInitialCondition.attitude_quaternion = transforms3d.quaternions.mat2quat(R).tolist()
    QuadInitialCondition.angular_velocity = [0, 0, 0]

    # define the desired goal
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    QuadDesiredStates = QuadStates()
    QuadDesiredStates.position = [2.5, 1.0, 1.5]
    QuadDesiredStates.velocity = [0, 0, 0]
    QuadDesiredStates.attitude_quaternion = transforms3d.quaternions.mat2quat(R).tolist()
    QuadDesiredStates.angular_velocity = [0, 0, 0]

    # run this method to obtain human inputs
    # SparseInput is an instance of dataclass DemoSparse
    Input = InputWaypoints(config_data)
    SparseInput = Input.run(QuadInitialCondition, QuadDesiredStates, ObsList)

    # create the quadrotor algorithm solver
    Solver = QuadAlgorithm(config_data, QuadParaInput, n_grid)

    # load the optimization method for learning iteration
    para_vanilla = {"learning_rate": 0.01, "iter_num": 5, "method": "Vanilla"} # This is for Vanilla gradient descent
    para_nesterov = {"learning_rate": 0.01, "iter_num": 5, "method": "Nesterov", "mu": 0.9, "true_loss_print_flag": True} # This is for Nesterov
    para_adam = {"learning_rate": 0.01, "iter_num": 5, "method": "Adam", "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8} # This is for Adam
    para_nadam = {"learning_rate": 0.01, "iter_num": 5, "method": "Nadam", "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8} # This is for Nadam
    
    loss_trace_comparison = []
    label_list = []
    # Vanilla gradient descent
    Solver.load_optimization_function(para_vanilla)
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, ObsList, print_flag=True, save_flag=False)
    loss_trace_vanilla = copy.deepcopy(Solver.loss_trace)
    loss_trace_comparison.append(loss_trace_vanilla)
    label_list.append(self.optimization_method_str)

    # Nesterov
    Solver.load_optimization_function(para_nesterov)
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, ObsList, print_flag=True, save_flag=False)
    loss_trace_nesterov = copy.deepcopy(Solver.loss_trace)
    loss_trace_comparison.append(loss_trace_nesterov)
    label_list.append(self.optimization_method_str)

    # Adam
    Solver.load_optimization_function(para_adam)
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, ObsList, print_flag=True, save_flag=False)
    loss_trace_adam = copy.deepcopy(Solver.loss_trace)
    loss_trace_comparison.append(loss_trace_adam)
    label_list.append(self.optimization_method_str)

    # Nadam
    Solver.load_optimization_function(para_nadam)
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, ObsList, print_flag=True, save_flag=False)
    loss_trace_nadam = copy.deepcopy(Solver.loss_trace)
    loss_trace_comparison.append(loss_trace_nadam, label_list)
    label_list.append(self.optimization_method_str)

    # plot the comparison
    Solver.plot_opt_method_comparison(loss_trace_comparison, label_list)
    
