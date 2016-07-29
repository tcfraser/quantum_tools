"""
Methods used to take a set of states and measurements and determine a probability distribution from it.
"""
from __future__ import print_function, division
import numpy as np
from ..statistics.probability import ProbDist
from ..utilities import utils
from .measurement import Measurement, ProjectiveMeasurement
from ..statistics.variable import RandomVariableCollection
from .state import State
from .. import config

class QuantumContext():

    def __init__(self, random_variables, measurements, states, permutation=None):
        self.random_variables = random_variables
        self.measurements = measurements
        self.states = states
        self.permutation = permutation
        self.num_measurements = len(measurements)
        self.num_states = len(states)
        
    def __str__(self):
        print_list = []
        print_list.append("QuantumContext: {0} measurements, {1} states.".format(self.num_measurements, self.num_states))
        for m in self.measurements:
            print_list += m._get_print_list()
        for s in self.states:
            print_list += s._get_print_list()
        return '\n'.join(print_list)

def QuantumProbDist(qc):
    joint_state = utils.tensor(*tuple(s.data for s in qc.states))
    if qc.permutation is not None:
        joint_state = utils.multidot(qc.permutation.T, joint_state, qc.permutation)
    def pdf(*args):
        measurement_operators = [qc.measurements[posn][val] for posn, val in enumerate(args)]
        joint_measurement = utils.tensor(*measurement_operators)

        p = np.trace(utils.multidot(joint_state, joint_measurement))
        assert(utils.is_small(p.imag)), "Probability is not real. It is {0}.".format(p)
        return p.real

    pd = ProbDist.from_callable_support(qc.random_variables, pdf)
    return pd

def QuantumProbDistOptimized(qc):
    """
    Only works for projective measurements
    """
    is_projective = [isinstance(m, ProjectiveMeasurement) for m in qc.measurements]
    assert(all(is_projective)), "measurements are not projective: {}".format(str(is_projective))
    despectral = [np.array(m.projectors).T for m in qc.measurements] # A*, B*, C*
    cum_measure_operators = utils.tensor(*despectral) # A* x B* x C*
    if qc.permutation is not None:
        cum_measure_operators = np.dot(qc.permutation, cum_measure_operators)
    joint_state = utils.tensor(*tuple(s.data for s in qc.states))
    super_support = utils.multidot(cum_measure_operators.conj().T, joint_state, cum_measure_operators)
    super_support_diag = np.diagonal(super_support)
    super_support_lookup = super_support_diag.reshape(qc.random_variables.outcome_space.get_input_base())
    support = np.copy(np.real(super_support_lookup))
    
    pd = ProbDist(qc.random_variables, support)
    return pd
