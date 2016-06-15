from ..statistics.probability import ProbDist
from ..statistics.variable import RandomVariableCollection
from ..utilities import utils
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities.constants import *
from ..utilities.utils import ei
import numpy as np

def print_symmetry_dist():
    pi = np.pi
    angle_spread = np.linspace(0, 2*pi, 24+1)
    cos_as = np.cos(angle_spread)
    sin_as = np.sin(angle_spread)
    states = np.vstack((cos_as, sin_as)).T
    dms = [utils.ket_to_dm(state) for state in states]
    dms = [utils.tensor(dm, dm) for dm in dms]
    A = Measurement([dms[0], dms[6]])
    rhoAB = State(dms[7])
    B = Measurement([dms[8], dms[14]])
    rhoBC = State(dms[15])
    C = Measurement([dms[16], dms[22]])
    rhoAC = State(dms[23])
    rvc = RandomVariableCollection.new(names=['A', 'B', 'C'], outcomes=[2,2,2])
    perm = utils.get_triangle_permutation()

    qc = QuantumContext(
        random_variables=rvc,
        measurements=(A,B,C),
        states=(rhoAB, rhoBC, rhoAC),
        permutation=perm)
    pd = QuantumProbDist(qc)
    print(pd)

if __name__ == '__main__':
    print_symmetry_dist()
