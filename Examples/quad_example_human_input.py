#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import json
import numpy as np
import transforms3d
from dataclasses import dataclass, field
from QuadPara import QuadPara
from QuadStates import QuadStates
from DemoSparse import DemoSparse
from QuadAlgorithm import QuadAlgorithm
from InputWaypoints import InputWaypoints


if __name__ == "__main__":
    # a json configuration file
    config_file_name = "config.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # define the quadrotor dynamics parameters
    QuadParaInput = QuadPara(inertial_list=[1.0, 1.0, 1.0], mass=1.0, l=1.0, c=0.02)

    # the learning rate
    learning_rate = 5e-3
    # the maximum iteration steps
    iter_num = 3
    # number of grids for nonlinear programming solver
    n_grid = 25

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

    # run this method to obtain human inputs
    # SparseInput is an instance of dataclass DemoSparse
    Input = InputWaypoints(config_data)
    SparseInput= Input.run(QuadInitialCondition, QuadDesiredStates)

    # create the quadrotor algorithm solver
    Solver = QuadAlgorithm(QuadParaInput, learning_rate, iter_num, n_grid)

    # solve it
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, print_flag=True, save_flag=True)

    print("waypoints")
    print(SparseInput.waypoints)
    print("time_list")
    print(SparseInput.time_list)
    print("time_horizon")
    print(SparseInput.time_horizon)