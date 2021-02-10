#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from QuadStates import QuadStates
from DemoSparse import DemoSparse
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
    num_obs = 8 # number of obstacles
    size_list = [0.2, 0.3, 0.4] # size lenth, width, height in x,y,z axis
    ObsList = generate_random_obs(num_obs, size_list, config_data)

    # define the initial condition
    QuadInitialCondition = QuadStates()
    QuadInitialCondition.position = [-2.0, -1.0, 0.6]

    # define the desired goal
    QuadDesiredStates = QuadStates()
    QuadDesiredStates.position = [2.5, 1.0, 1.5]

    # initialize the class
    Input = InputWaypoints(config_data)

    # run this method to obtain human inputs
    # SparseInput is an instance of dataclass DemoSparse
    # waypoints_output is a 2D list, each sub-list is a waypoint [x, y, z], not including the start and goal
    # time_list_all is a 1D list to store the time-stamp for each waypoint, including the start and goal
    SparseInput = Input.run(QuadInitialCondition, QuadDesiredStates, ObsList)

    print("waypoints")
    print(SparseInput.waypoints)
    print("time_list")
    print(SparseInput.time_list)
    print("time_horizon")
    print(SparseInput.time_horizon)