"""
Generators of quantum states
"""
import numpy as np
from constants import *
from utils import Utils
import global_config

class State():

    def __init__(self, data):
        self.data = data

    def __str__(self):
        print_list = []
        print_list.append(self.__repr__())
        print_list.append(str(self.data))
        return '\n'.join(print_list)

    @staticmethod
    def dm(t):
        g = Utils.cholesky(t)
        g /= (np.trace(g) + mach_eps)
        # assert(is_trace_one(groundn)), "Trace of g is not 1.0! Difference: {0}".format(np.trace(g) - 1)
        rho = State(g)
        return rho