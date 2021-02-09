#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/lib')
import random
from dataclasses import dataclass, field
from ObsInfo import ObsInfo


def generate_random_obs(num_obs: int, size_list: list, config_data):
    """
    config_file_name = "config.json"
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    size_list = [length, width, height]

    """

    ObsList = []

    if (num_obs > 0.5):
        for i in range(0, num_obs):
            # random center
            center = [random.uniform(config_data["LAB_SPACE_LIMIT"]["LIMIT_X"][0], config_data["LAB_SPACE_LIMIT"]["LIMIT_X"][1]), \
                random.uniform(config_data["LAB_SPACE_LIMIT"]["LIMIT_Y"][0], config_data["LAB_SPACE_LIMIT"]["LIMIT_Y"][1]), \
                random.uniform(config_data["LAB_SPACE_LIMIT"]["LIMIT_Z"][0], config_data["LAB_SPACE_LIMIT"]["LIMIT_Z"][1])]

            ObsList.append( ObsInfo(center, size_list) )

    return ObsList