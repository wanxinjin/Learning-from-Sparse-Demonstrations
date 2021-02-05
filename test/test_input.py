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


if __name__ == "__main__":

    # a json configuration file
    config_file_name = "config.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # define the initial condition
    QuadInitialCondition = QuadStates()
    QuadInitialCondition.position = [-2.0, -1.0, 0.6]

    # define the desired goal
    QuadDesiredStates = QuadStates()
    QuadDesiredStates.position = [2.5, 1.0, 1.5]

    # initialize the class
    Input = InputWaypoints(config_data)

    # run this method to obtain human inputs
    # waypoints_output is a 2D list, each sub-list is a waypoint [x, y, z], not including the start and goal
    # time_list_all is a 1D list to store the time-stamp for each waypoint, including the start and goal
    waypoints_output, time_list_all = Input.run(QuadInitialCondition, QuadDesiredStates)

    # define the sparse demonstration
    SparseInput = DemoSparse()
    SparseInput.waypoints = waypoints_output
    SparseInput.time_list = time_list_all[1 : -1]
    SparseInput.time_horizon = time_list_all[-1]

    print("waypoints")
    print(SparseInput.waypoints)
    print("time_list")
    print(SparseInput.time_list)
    print("time_horizon")
    print(SparseInput.time_horizon)