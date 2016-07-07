"""
Contains methods used for generating and interacting with generic measurements.
"""
from __future__ import print_function, division
import numpy as np
from scipy import linalg
from ..utilities import utils
from ..utilities import rmt
import random

class Measurement():

    def __init__(self, operators):
        self._operators = [np.matrix(o) for o in operators]
        # The size of the matrix associated with this measurement.
        self._size = self._operators[0].shape[0]
        # assertions
        # print(sum(self._operators), np.eye(self._size))
        assert(np.allclose(sum(self._operators), np.eye(self._size))), "Measurement Operators don't sum to the identity."
        for o in self._operators:
            assert(utils.is_psd(o)), "Operator is not positive semi-definite."
            assert(utils.is_hermitian(o)), "Operator is not hermitian."

    def __getitem__(self, key):
        return self._operators[key]

    def __iter__(self):
        for i in self._operators:
            yield i

    def __str__(self):
        print_list = []
        print_list.append(self.__repr__())
        print_list.append("Number of Outcomes: {0}".format(self.num_outcomes))
        print_list.append("Size of Operators: {0}".format(self._size))
        print_list.append("Operators:")
        for operator in self._operators:
            print_list.append(str(operator))
        return '\n'.join(print_list)

class MeasurementStrats():
    # Namespace Declarations
    pass

class MeasurementStratsRandom():

    @staticmethod
    def pvms_uniform(size):
        u = rmt.U(size)
        S = [utils.ket_to_dm(u[:,i]) for i in range(size)]
        random.shuffle(S)
        m = Measurement(S)
        return m

    @staticmethod
    def proj_comp(size):
        matrices = []
        for i in range(size):
            q = np.zeros((size, size))
            q[i, i] = 1
            matrices.append(q)
        m = Measurement(matrices)
        return m

    @staticmethod
    def seesaw(dim, count):
        S = rmt.P_I(dim, count)
        unitary = rmt.U(dim)
        unitaryt = unitary.conj().T
        for i in range(count):
            S[i] = utils.multidot(unitary, S[i], unitaryt)
        m = Measurement(S)
        return m

class MeasurementStratsDeterministic():

    @staticmethod
    def deterministic(size):
        I = np.identity(4)
        matrices = [I]
        for i in range(size - 1):
            matrices.append(np.zeros((size, size)))
        m = Measurement(matrices)
        return m

class MeasurementStratsParam():

    @staticmethod
    def pvms_older(param):
        g = utils.cholesky(param)
        eigen_values, eigen_vectors = linalg.eigh(g)
        density_matrices = [utils.ket_to_dm(eigen_vectors[:,i]) for i in range(eigen_values.shape[0])]
        m = Measurement(density_matrices)
        return m

    @staticmethod
    def seeded_pvms(param, seed_operators):
        GL_C = utils.param_GL_C(param)
        U = rmt.GL_knit_QR(GL_C)
        Ut = U.conj().T
        operators = [utils.multidot(Ut, o, U) for o in seed_operators]
        m = Measurement(operators)
        return m

    @staticmethod
    def pvms_outcomes(num_outcomes, param):
        GL_C = utils.param_GL_C(param)
        U = rmt.GL_knit_QR(GL_C)
        Ut = U.conj().T
        projectors = utils.projectors(num_outcomes)
        operators = [utils.multidot(Ut, o, U) for o in projectors]
        m = Measurement(operators)
        return m

    @staticmethod
    def pvms(param):
        GL_C = utils.param_GL_C(param)
        U = rmt.GL_knit_QR(GL_C)
        operators = [utils.ket_to_dm(U[:, i]) for i in range(U.shape[1])]
        m = Measurement(operators)
        return m

    @staticmethod
    def povms(param, outcomes):
        # Currently Broken
        assert(False), "Currently Broken: povms are made with necessary but not sufficient conditions."
        lp = len(param)
        block_size = int(lp / (outcomes - 1))
        size = int(block_size**(1/2))
        povms = []
        largest_eig = []
        for i in range(outcomes - 1):
            povm_i = utils.cholesky(param[i*block_size:(i+1)*block_size])
            largest_eig = utils.largest_eig(povm_i)
            povms.append(povm_i / largest_eig)
        I = np.eye(size)
        povms.append(I - sum(povms))
        m = Measurement(povms)
        return m

# === Scope Declarations ===
Measurement.Strats = MeasurementStrats
MeasurementStrats.Random = MeasurementStratsRandom
MeasurementStrats.Param = MeasurementStratsParam
MeasurementStrats.Deterministic = MeasurementStratsDeterministic