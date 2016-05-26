"""
Contains methods used for generating and interacting with generic measurements.
"""
from utils import Utils
from scipy import linalg
import numpy as np
import global_config

class Measurement():

    def __init__(self, operators):
        self._operators = operators
        self.num_outcomes = len(self._operators)
        # The size of the matrix associated with this measurement.
        self._size = self._operators[0].shape[0]

    def __getitem__(self, key):
        return self._operators[key]

    def __str__(self):
        print_list = []
        print_list.append(self.__repr__())
        print_list.append("Number of Outcomes: {0}".format(self.num_outcomes))
        print_list.append("Size of Operators: {0}".format(self._size))
        print_list.append("Operators:")
        for operator in self._operators:
            print_list.append(str(operator))
        return '\n'.join(print_list)

    @staticmethod
    def pvms(t):
        g = Utils.cholesky(t)
        eigen_values, eigen_vectors = linalg.eigh(g)
        density_matrices = [Utils.ket_to_dm(eigen_vectors[:,i]) for i in range(eigen_values.shape[0])]
        m = Measurement(density_matrices)
        return m

    @staticmethod
    def povms(t, number):
        pass
