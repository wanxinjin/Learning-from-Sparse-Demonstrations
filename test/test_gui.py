#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from MplWidget import MplWidget
from DemoSparse import DemoSparse
from QuadStates import QuadStates
from ObsInfo import ObsInfo
from generate_random_obs import generate_random_obs


if __name__ == "__main__":

    # a json configuration file
    config_file_name = "config.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    # generate random obstacles
    num_obs = 30 # number of obstacles
    size_list = [0.2, 0.3, 0.4] # size lenth, width, height in x,y,z axis
    ObsList = generate_random_obs(num_obs, size_list, config_data)


    app = QtWidgets.QApplication(sys.argv)
    w = MplWidget(ObsList, config_data)
    w.show()
    sys.exit(app.exec_())
