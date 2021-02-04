#!/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class QuadPara:
    inertial_x: float = 1 # [kg*m^2] the moment of inertia in x-axis
    inertial_y: float = 1 # [kg*m^2] the moment of inertia in y-axis
    inertial_z: float = 1 # [kg*m^2] the moment of inertia in z-axis
    mass: float = 1 # [kg] the mass of quadrotor
    l: float = 1 # [meter] the length from quadrotor center of mass to each rotor center
    c: float = 1 # rotor parameter