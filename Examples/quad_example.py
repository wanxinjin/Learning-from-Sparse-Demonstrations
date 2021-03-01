#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
sys.path.append('/Users/zehui/Downloads/casadi-osx-py39-v3.5.5')
import json
import numpy as np
import transforms3d
from dataclasses import dataclass, field
from QuadPara import QuadPara
from QuadStates import QuadStates
from DemoSparse import DemoSparse
from QuadAlgorithm import QuadAlgorithm


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # define the quadrotor dynamics parameters
    QuadParaInput = QuadPara(inertial_list=[1.0, 1.0, 1.0], mass=1.0, l=1.0, c=0.02)

    # number of grids for nonlinear programming solver
    n_grid = 25

    # define the sparse demonstration
    SparseInput = DemoSparse()
    SparseInput.waypoints = [
        [0.5, 0.5, 0.6],
        [1.0, 1.0, 0.8],
        [1.5, 1.5, 1.0],
        [2.0, 2.0, 1.2],
        [2.5, 2.5, 1.5]]
    SparseInput.time_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    SparseInput.time_horizon = 6

    # define the initial condition
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    QuadInitialCondition = QuadStates()
    QuadInitialCondition.position = [0, 0, 0.6]
    QuadInitialCondition.velocity = [0, 0, 0]
    QuadInitialCondition.attitude_quaternion = transforms3d.quaternions.mat2quat(R).tolist()
    QuadInitialCondition.angular_velocity = [0, 0, 0]

    # define the desired goal
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    QuadDesiredStates = QuadStates()
    QuadDesiredStates.position = [3, 3, 1.5]
    QuadDesiredStates.velocity = [0, 0, 0]
    QuadDesiredStates.attitude_quaternion = transforms3d.quaternions.mat2quat(R).tolist()
    QuadDesiredStates.angular_velocity = [0, 0, 0]

    # create the quadrotor algorithm solver
    Solver = QuadAlgorithm(config_data, QuadParaInput, n_grid)

    # load the optimization method for learning iteration
    # para_optimization_dict = {"learning_rate": 0.01, "iter_num": 10, "method": "Vanilla"} # This is for Vanilla gradient descent
    para_optimization_dict = {"learning_rate": 0.01, "iter_num": 10, "method": "Nesterov", "mu": 0.9} # This is for Nesterov Momentum
    Solver.load_optimization_method(para_optimization_dict)

    # solve it
    # method_string: "Vanilla" or "Nesterov"
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, ObsList=[], print_flag=True, save_flag=True)

