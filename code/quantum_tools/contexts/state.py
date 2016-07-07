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
        assert(utils.is_trace_one(self.data)), "State does not have unitary trace."
        assert(utils.is_psd(self.data)), "State is not positive semi-definite."
        assert(utils.is_hermitian(self.data)), "State is not hermitian."

    def __str__(self):
        print_list = []
        print_list.append(self.__repr__())
        print_list.append(str(self.data))
        return '\n'.join(print_list)

class StateStrats():
    pass

class StateStratsParam():

    @staticmethod
    def dm(t):
        g = utils.cholesky(t)
        g /= (np.trace(g) + mach_eps)
        rho = State(g)
        return rho

class StateStratsDeterministic():

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

class StateStratsRandom():

    @staticmethod
    def pure_uniform(n):
        """
            n = 2 is qubits
        """
        bloch_sample = utils.uniform_n_sphere_metric(n)
        phases = utils.uniform_phase_components(n)
        psi = bloch_sample * phases
        dm = utils.ket_to_dm(psi)
        rho = State(dm)
        return rho

# === Scope Declarations ===
State.Strats = StateStrats
StateStrats.Random = StateStratsRandom
StateStrats.Param = StateStratsParam
StateStrats.Deterministic = StateStratsDeterministic