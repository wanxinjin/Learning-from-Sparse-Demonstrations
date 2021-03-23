#!/usr/bin/env python3
from dataclasses import dataclass, field


@dataclass
class QuadStates:
    # [meter] the position
    position: list = field(default_factory=lambda: [0, 0, 0])
    # [m/s] the velocity
    velocity: list = field(default_factory=lambda: [0, 0, 0])
    # the attitude in quaternion
    attitude_quaternion: list = field(default_factory=lambda: [1, 0, 0, 0])
    # [1/sec] the angular velocity
    angular_velocity: list = field(default_factory=lambda: [0, 0, 0])
