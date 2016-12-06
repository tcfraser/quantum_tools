"""
Generators of quantum states
"""
from __future__ import print_function, division
import numpy as np
from ..utilities.constants import *
from ..rmt.utils import *
from ..utilities import utils
from ..config import *

class State():

    def __init__(self, data):
        self.data = np.matrix(data)
        if ASSERT_MODE >= 1:
            assert(utils.is_trace_one(self.data)), "State does not have unitary trace."
            assert(utils.is_psd(self.data)), "State is not positive semi-definite."
            assert(utils.is_hermitian(self.data)), "State is not hermitian."

    def __str__(self):
        return '\n'.join(self._get_print_list())

    def _get_print_list(self):
        print_list = []
        print_list.append(self.__repr__())
        print_list.append(str(self.data))
        return print_list

class StateStrats():
    pass

class StateStratsParam():

    @staticmethod
    def dm(t):
        g = cholesky(t)
        g /= (np.trace(g) + mach_eps)
        rho = State(g)
        return rho

class StateStratsDeterministic():

    @staticmethod
    def maximally_entangled_bell(n=0):
        """
        Maximally Entangled Bell State
        """
        n = n % 4
        rho = State(utils.ket_to_dm(mebs[:,n]))
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