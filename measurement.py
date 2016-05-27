"""
Contains methods used for generating and interacting with generic measurements.
"""
from __future__ import print_function, division
from utils import Utils
from scipy import linalg
import numpy as np
from pprint import pprint
import global_config

class Measurement():

    def __init__(self, operators):
        self._operators = [np.matrix(o) for o in operators]
        self.num_outcomes = len(self._operators)
        # The size of the matrix associated with this measurement.
        self._size = self._operators[0].shape[0]

    def __getitem__(self, key):
        return self._operators[key]

    # def __iter__(self):
    #     return self._operators

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


    # @staticmethod
    # def sbs(t, num_outcomes):
    #     """
    #     Seperable bloch sphere measurements
    #     Requires num_outcomes * 2 * num_qubits param
    #     """
    #     measures = []
    #     num_param = len(t)
    #     j = 0
    #     sub_measures_per_outcome = num_param // 2 // num_outcomes
    #     while j < num_param:
    #         separated_measures = (Utils.get_meas_on_bloch_sphere(*t[j + 2*i: j + 2*(i+1)]) for i in range(sub_measures_per_outcome))
    #         joint_measurement = Utils.tensor(*tuple(separated_measures))
    #         measures.append(joint_measurement)
    #         j += sub_measures_per_outcome*2
    #     m = Measurement(measures)
    #     return m

    @staticmethod
    def sbs(t):
        """
        Seperable bloch sphere measurements
        """
        basis_pairs_args = [t[2*(i):2*(i+1)] for i in range(len(t)//2)]
        # print(basis_pairs_args)
        basis_pairs = list(map(Utils.get_orthogonal_pair, basis_pairs_args))
        pprint(basis_pairs)
        # print(basis_pairs)
        # for bp in basis_pairs:
        #     print(bp[0] + bp[1])
        configuration = [
            [[0, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 2]],
            [[1, 0], [1, 2]],
        ]
        measures = []
        for config in configuration:
            m_1 = basis_pairs[config[0][1]][config[0][0]]
            m_2 = basis_pairs[config[1][1]][config[1][0]]
            joint_measurement = Utils.tensor(m_1, m_2)
            measures.append(joint_measurement)
        m = Measurement(measures)
        return m

def perform_tests():
    m = Measurement.sbs(np.random.random(6))
    for mi in m:
        print(Utils.is_psd(mi))
    print(sum(m._operators))
    print(m)

if __name__ == '__main__':
    perform_tests()