#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import numpy as np
import transforms3d
from dataclasses import dataclass, field
from QuadPara import QuadPara
from QuadStates import QuadStates
from DemoSparse import DemoSparse
from QuadAlgorithm import QuadAlgorithm


if __name__ == "__main__":
    # define the quadrotor dynamics parameters
    QuadParaInput = QuadPara()
    QuadParaInput.inertial_x = 1.0
    QuadParaInput.inertial_y = 1.0
    QuadParaInput.inertial_z = 1.0
    QuadParaInput.mass = 1.0
    QuadParaInput.l = 1.0
    QuadParaInput.c = 0.02

    # the learning rate
    learning_rate = 5e-3
    # the maximum iteration steps
    iter_num = 5
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
    SparseInput.time_lists = [1.0, 2.0, 3.0, 4.0, 5.0]
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
    Solver = QuadAlgorithm(QuadParaInput, learning_rate, iter_num, n_grid)

    # solve it
    Solver.run(QuadInitialCondition, QuadDesiredStates, SparseInput, print_flag=True, save_flag=True)

