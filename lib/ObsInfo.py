#!/usr/bin/env python3
from dataclasses import dataclass, field


@dataclass
class ObsInfo:
    length: float = 1 # [meter] the length (x-axis) of the obstacle
    width: float = 1 # [meter] the width (y-axis) of the obstacle
    height: float = 1 # [meter] the height (z-axis) of the obstacle
    # [meter] the [x, y, z] coordinates of the center of the obstacle
    center: list = field(default_factory=lambda: [0, 0, 0])


    def __init__(self, center_pisition: list, size_list: list):
        self.center = center_pisition
        self.length = size_list[0]
        self.width = size_list[1]
        self.height = size_list[2]
