from . import marginal_equality
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities import utils
from ..utilities.profiler import profile
from ..statistics.variable import RandomVariableCollection
from . import positive_linear_solve

perm = utils.get_triangle_permutation()

def uniform_sample_qdistro(rvc):
    A = Measurement.Strats.Random.pvms_uniform(4)
    B = Measurement.Strats.Random.pvms_uniform(4)
    C = Measurement.Strats.Random.pvms_uniform(4)
    rhoAB = State.Strats.Random.pure_uniform(4)
    rhoBC = State.Strats.Random.pure_uniform(4)
    rhoAC = State.Strats.Random.pure_uniform(4)
    # rhoAB = State.Strats.Deterministic.mebs(2)
    # rhoBC = State.Strats.Deterministic.mebs(2)
    # rhoAC = State.Strats.Deterministic.mebs(2)
    # print(A[4].shape)
    # print(rhoBC.shape)
    qc = QuantumContext(random_variables=rvc, measurements=(A,B,C), states=(rhoAB, rhoBC, rhoAC), permutation=perm)
    pd = QuantumProbDist(qc)
    return pd

@profile
def go():
    symbolic_contexts = [
        [['A2'], ['B2'], ['C2']],
        [['B2'], ['A2',   'C1']],
        [['C2'], ['A1',   'B2']],
        [['A2'], ['B1',   'C2']],
        [['A1',   'B1',   'C1']],
    ]
    inflation_rvc = RandomVariableCollection.new(names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], outcomes=[4,4,4,4,4,4])
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    defl_map = marginal_equality.get_delf_map(symbolic_contexts)
    pd = uniform_sample_qdistro(original_rvc)
    print(pd)

    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(pd, symbolic_contexts, defl_map)
    res = positive_linear_solve.get_feasibility_cvx(A, b)
    print(res)

if __name__ == '__main__':
    go()
