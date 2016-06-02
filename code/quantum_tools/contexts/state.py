"""
Generators of quantum states
"""
from __future__ import print_function, division
import numpy as np
from ..utilities.constants import *
from ..utilities import utils
from ..config import *

class State():

    def __init__(self, data):
        self.data = np.matrix(data)

    def __str__(self):
        print_list = []
        print_list.append(self.__repr__())
        print_list.append(str(self.data))
        return '\n'.join(print_list)

    @staticmethod
    def dm(t):
        g = utils.cholesky(t)
        g /= (np.trace(g) + mach_eps)
        # assert(is_trace_one(groundn)), "Trace of g is not 1.0! Difference: {0}".format(np.trace(g) - 1)
        rho = State(g)
        return rho

    @staticmethod
    def mebs(n=0):
        """
        Maximally Entangled Bell State
        """
        n = n % 4
        norm = 1/np.sqrt(2)
        if n == 0:
            psi = norm * (qb00 + qb11)
        elif n == 1:
            psi = norm * (qb00 - qb11)
        elif n == 2:
            psi = norm * (qb01 + qb10)
        elif n == 3:
            psi = norm * (qb01 - qb10)
        rho = State(utils.ket_to_dm(psi))
        return rho
