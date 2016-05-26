"""
Methods used to take a set of states and measurements and determine a probabilty distribution from it.
"""
from probabilty_distro import ProbDistro
from utils import Utils

class QuantumProbDistro(ProbDistro):

    def __init__(self, measurements, states, permutation=None):
        self._measurements = measurements
        self._num_measurements = len(self._measurements)
        self._states = states
        self._num_states = len(self._states)
        self._permutation = permutation
        self._num_outcomes_per_measurement = max(m.num_outcomes for m in self._measurements)

        super().__init__()

    def P(*args):
        num_args = len(args)
        if (len(args))
        self._num_outcomes_per_measurement