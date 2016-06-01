"""
Methods used to take a set of states and measurements and determine a probability distribution from it.
"""
from __future__ import print_function, division
from probability import ProbDist
from utils import Utils
import numpy as np
from measurement import Measurement
from variable import RandomVariableCollection
from state import State
from timing_profiler import timing
import global_config
from ineqs import *

class QuantumContext():

    def __init__(self, measurements, states, permutation=None):
        self.measurements = RandomVariableCollection(measurements)
        self.states = states
        self.permutation = permutation
        self.permutationT = permutation.T if permutation is not None else None
        self.num_measurements = len(measurements)
        self.num_states = len(states)
        self.num_outcomes_per_measurement = max(m.num_outcomes for m in measurements)

def QuantumProbDist(qc):
    def pdf(*args):
        measurement_operators = [qc.measurements[posn][val] for posn, val in enumerate(args)]
        joint_measurement = Utils.tensor(*measurement_operators)
        joint_state = Utils.tensor(*tuple(s.data for s in qc.states))
        if qc.permutation is not None:
            joint_state = Utils.multiply(qc.permutationT, joint_state, qc.permutation)

        p = np.trace(Utils.multiply(joint_state, joint_measurement))
        assert(Utils.is_small(p.imag))
        return p.real

    pd = ProbDist.from_callable_support(qc.measurements, pdf)
    return pd

def gen_test_distribution():
    perm = Utils.get_permutation()
    A = Measurement.pvms('A', np.random.random(16))
    B = Measurement.pvms('B', np.random.random(16))
    C = Measurement.pvms('C', np.random.random(16))
    # A = Measurement.proj_comp('A',4)
    # B = Measurement.proj_comp('B',4)
    # C = Measurement.proj_comp('C',4)
    # A = Measurement.deterministic('A',4)
    # B = Measurement.deterministic('B',4)
    # C = Measurement.deterministic('C',4)
    # A = Measurement.seesaw('A',4)
    # B = Measurement.seesaw('B',4)
    # C = Measurement.seesaw('C',4)
    # A = Measurement.unitary_pvms('A',4)
    # B = Measurement.unitary_pvms('B',4)
    # C = Measurement.unitary_pvms('C',4)
    rhoAB = State.dm(np.random.random(16))
    rhoAC = State.dm(np.random.random(16))
    rhoBC = State.dm(np.random.random(16))

    qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=perm)
    qpd = QuantumProbDist(qc)
    return qpd

def two_outcome_triangle():
    perm = Utils.get_permutation()
    A = Measurement.povms('A', np.random.normal(0,1,16), 2)
    B = Measurement.povms('B', np.random.normal(0,1,16), 2)
    C = Measurement.povms('C', np.random.normal(0,1,16), 2)
    # print(A)
    # A = Measurement.proj_comp('A',4)
    # B = Measurement.proj_comp('B',4)
    # C = Measurement.proj_comp('C',4)
    # A = Measurement.deterministic('A',4)
    # B = Measurement.deterministic('B',4)
    # C = Measurement.deterministic('C',4)
    # A = Measurement.seesaw('A',4)
    # B = Measurement.seesaw('B',4)
    # C = Measurement.seesaw('C',4)
    # A = Measurement.unitary_pvms('A',4)
    # B = Measurement.unitary_pvms('B',4)
    # C = Measurement.unitary_pvms('C',4)
    rhoAB = State.dm(np.random.normal(0,1,16))
    rhoAC = State.dm(np.random.normal(0,1,16))
    rhoBC = State.dm(np.random.normal(0,1,16))


    qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=perm)
    qpd = QuantumProbDist(qc)
    return qpd

@timing
def perform_tests():
    # print(Measurement.pvms(np.random.random(16), 4))
    # print(Measurement.pvms(np.random.random(4), 2))
    # print(State.dm(np.random.random(16), 4))
    # rhoAB = State.dm(np.random.random(16))
    # rhoBC = State.dm(np.random.random(16))
    # rhoAC = State.dm(np.random.random(16))
    # perm = Utils.get_permutation()

    # qc = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=perm)
    # qpd = QuantumProbDist(qc)
    # print(qpd)
    # print(mutual_information(qpd, 2, 1))
    # print(mutual_information(qpd, 2, 0))
    # print(entropy(qpd, (1,0)))
    # print(entropy(qpd, (1,2)))
    # qpdfritz = gen_test_distribution()
    qpdfritz = two_outcome_triangle()
    print(qpdfritz.prob({}))
    # IAB = qpdfritz.mutual_information(['A', 'B'])
    # IAC = qpdfritz.mutual_information(['A', 'C'])
    # HA = qpdfritz.entropy('A')

    # print(IAB, IAC, HA, HA - IAC - IAB)
    print(HLP1(qpdfritz))
    print(HLP2(qpdfritz))
    print(HLP3(qpdfritz))


if __name__ == '__main__':
    perform_tests()