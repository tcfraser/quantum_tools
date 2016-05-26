"""
Methods used to take a set of states and measurements and determine a probabilty distribution from it.
"""
from probability_distro import ProbDistro, mutual_information, entropy
from utils import Utils
import numpy as np
from measurement import Measurement
from state import State
from timing_profiler import timing
import global_config

class QuantumContext():

    def __init__(self, measurements, states, permutation=None):
        self.measurements = measurements
        self.states = states
        self.permutation = permutation
        self.permutationT = permutation.T if permutation is not None else None
        self.num_measurements = len(measurements)
        self.num_states = len(states)
        self.num_outcomes_per_measurement = max(m.num_outcomes for m in measurements)

def QuantumProbDistro(qc):
    def pdf(*args):
        measurement_operators = [qc.measurements[posn][val] for posn, val in enumerate(args)]
        joint_measurement = Utils.tensor(*measurement_operators)
        joint_state = Utils.tensor(*tuple(s.data for s in qc.states))
        if qc.permutation is not None:
            joint_state = Utils.multiply(qc.permutationT, joint_state, qc.permutation)

        p = np.trace(Utils.multiply(joint_state, joint_measurement))
        assert(Utils.is_small(p.imag))
        return p.real

    pd = ProbDistro.from_callable_support(pdf, num_outcomes=qc.num_outcomes_per_measurement, num_variables=qc.num_measurements)
    return pd

@timing
def perform_tests():
    # print(Measurement.pvms(np.random.random(16), 4))
    # print(Measurement.pvms(np.random.random(4), 2))
    # print(State.dm(np.random.random(16), 4))
    A = Measurement.pvms(np.random.random(16))
    B = Measurement.pvms(np.random.random(16))
    C = Measurement.pvms(np.random.random(16))
    rhoAB = State.dm(np.random.random(16))
    rhoBC = State.dm(np.random.random(16))
    rhoAC = State.dm(np.random.random(16))
    perm = Utils.get_permutation()

    qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=perm)
    qpd = QuantumProbDistro(qc)
    print(qpd)
    print(mutual_information(qpd, 2, 1))
    print(mutual_information(qpd, 2, 0))
    print(entropy(qpd, (1,0)))
    # print(entropy(qpd, (1,2)))

if __name__ == '__main__':
    perform_tests()