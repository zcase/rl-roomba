# To give credit where credit is do, this code is based off of the works found at the following sites:
#   https://github.com/raykking/DQN_lidar
#   This is part of the code that was provided by paper [3] Dynamic Path Planning of Unknown Environment Based on Deep Reinforcement Learning

from math import sqrt, cos, sin, atan2
import numpy as np
import random


class Vector(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def cal_vector(self, p1, p2):
        self.x = p2[0] - p1[0]
        self.y = p2[1] - p1[1]
        return self.x, self.y

    def get_magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)
