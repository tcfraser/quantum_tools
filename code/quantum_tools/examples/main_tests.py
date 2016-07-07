from ..config import *
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities.timing_profiler import timing
from ..statistics.variable import RandomVariableCollection
from ..examples.ineqs import *
from ..utilities import utils
import numpy as np

def save_file_to_output():
    with open(OUTPUT_DIR + 'test.txt', 'w+') as f:
        f.write('This is a successful test.')
    print("Wrote to file.")

def gen_test_distribution():
    perm = utils.get_permutation()
    rvc = RandomVariableCollection.new(names=['A', 'B', 'C'], outcomes=[4,4,4])

    A = Measurement.Strats.Param.pvms(num_outcomes=4, param=np.random.normal(0,1,16*2))
    B = Measurement.Strats.Param.pvms(num_outcomes=4, param=np.random.normal(0,1,16*2))
    C = Measurement.Strats.Param.pvms(num_outcomes=4, param=np.random.normal(0,1,16*2))
    rhoAB = State.dm(np.random.normal(0,1,16))
    rhoAC = State.dm(np.random.normal(0,1,16))
    rhoBC = State.dm(np.random.normal(0,1,16))

    qc = QuantumContext(random_variables=rvc, measurements=(A,B,C), states=(rhoAB,rhoBC,rhoAC), permutation=perm)
    qpd = QuantumProbDist(qc)
    return qpd

@timing
def context_tests():
    tpd = gen_test_distribution()
    print(tpd)
    print(tpd.prob({}))
    print(HLP1(tpd))
    print(HLP2(tpd))
    print(HLP3(tpd))

@timing
def print_tests():
    print(utils.param_GL_C(np.random.normal(0,1,32)))
    return
    m = Measurement.Strats.Deterministic.deterministic('A', 4)
    print(m)
    m = Measurement.Strats.Random.seesaw('A', 4)
    print(m)
    m = Measurement.Strats.Random.pvms('A', 4)
    print(m)
    print(Measurement)

if __name__ == '__main__':
    # print_tests()
    context_tests()
    # save_file_to_output()