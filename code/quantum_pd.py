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

class QuantumContext():

    def __init__(self, measurements, states, permutation=None):
        self.measurements = RandomVariableCollection(measurements)
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

    pd = ProbDist.from_callable_support(qc.measurements, pdf)
    return pd

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
    # qpd = QuantumProbDistro(qc)
    # print(qpd)
    # print(mutual_information(qpd, 2, 1))
    # print(mutual_information(qpd, 2, 0))
    # print(entropy(qpd, (1,0)))
    # print(entropy(qpd, (1,2)))

    # === Fritz ===
    perm = Utils.get_permutation()
    # A = Measurement.pvms('A', np.random.random(16))
    # B = Measurement.pvms('B', np.random.random(16))
    # C = Measurement.pvms('C', np.random.random(16))
    # A = Measurement.proj_comp('A',4)
    # B = Measurement.proj_comp('B',4)
    # C = Measurement.proj_comp('C',4)
    # A = Measurement.deterministic('A',4)
    # B = Measurement.deterministic('B',4)
    # C = Measurement.deterministic('C',4)
    A = Measurement.seesaw('A',4)
    B = Measurement.seesaw('B',4)
    C = Measurement.seesaw('C',4)
    rhoAB = State.dm(np.random.random(16))
    rhoAC = State.dm(np.random.random(16))
    rhoBC = State.dm(np.random.random(16))

    # A = B = Measurement.sbs([
    #     0, 0,
    #     np.pi/4, 0,
    #     3*np.pi/4, 0,
    #     ])

    # C = Measurement.sbs([
    #     0, 0,
    #     0, 0,
    #     0, 0,
    #     ])
    qcfritz = QuantumContext(measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=perm)
    qpdfritz = QuantumProbDistro(qcfritz)
    # IAB = qpdfritz.mutual_information(['A', 'B'])
    # IAC = qpdfritz.mutual_information(['A', 'C'])
    # HA = qpdfritz.entropy('A')

    # print(IAB, IAC, HA, HA - IAC - IAB)
    print(HLP1(qpdfritz))
    print(HLP2(qpdfritz))
    print(HLP3(qpdfritz))

def HLP3(PD):
    H = PD.H
    I = PD.I
    result = H(['A', 'B']) - I(['A', 'B', 'C']) + I(['A', 'B']) + I(['B', 'C']) + I(['C', 'A'])
    return result

def HLP2(PD):
    H = PD.H
    I = PD.I
    result = H('A') + H('B') + H('C') - 2 * (I(['A', 'B', 'C']) + I(['A', 'B']) + I(['B', 'C']) + I(['C', 'A']))
    return result

def HLP1(PD):
    H = PD.H
    I = PD.I
    result = H('A') - I(['A', 'B']) - I(['A', 'C'])
    return result

if __name__ == '__main__':
    perform_tests()