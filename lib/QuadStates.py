#!/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class QuadStates:
    position: list = [0, 0, 0] # [meter] the position
    velocity: list = [0, 0, 0] # [m/s] the velocity
    attitude_quaternion: list = [1, 0, 0, 0] # the attitude in quaternion
    angular_velocity: list = [0, 0, 0] # [1/sec] the angular velocity
