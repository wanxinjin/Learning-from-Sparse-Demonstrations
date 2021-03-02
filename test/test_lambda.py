#!/usr/bin/env python3
import os
import sys
import time
import numpy as np


class Test:

    def __init__(self, x, y, a):
        self.x = x
        self.y = y

        self.func = lambda self, a: (self.func111(x, y), self.func111(y, a))

        print(self.func(self, a))


    def func111(self, x, y):
        return x+y, x-y


if __name__ == "__main__":
    A = Test(5, 3, 100)
    print(A.x)
    print(A.y)


