#!/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class DemoSparse:
    # [meter] a 2D list of waypoints, each 1D list is a waypoint [px, py, pz]
    waypoints: list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # [sec] a 1D list of time (tau) in the waypoint frame
    time_list: list = [1, 2, 3]
    # [sec] total time horizon for these waypoints (T)
    time_horizon: float = 4
