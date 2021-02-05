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

    Input = InputWaypoints(config_data)
    waypoints_output = Input.run(QuadInitialCondition, QuadDesiredStates)
