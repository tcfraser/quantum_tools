from ..inflation import marginal_equality
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities import utils
from ..utilities.profiler import profile
from ..statistics.variable import RandomVariableCollection
from ..inflation import positive_linear_solve
from ..examples import prob_dists
from pprint import pprint

perm = utils.get_triangle_permutation()

def uniform_sample_qdistro(rvc):
    A = Measurement.Strats.Random.pvms_uniform(4)
    B = Measurement.Strats.Random.pvms_uniform(4)
    C = Measurement.Strats.Random.pvms_uniform(4)
    rhoAB = State.Strats.Random.pure_uniform(4)
    rhoBC = State.Strats.Random.pure_uniform(4)
    rhoAC = State.Strats.Random.pure_uniform(4)
    # rhoAB = State.Strats.Deterministic.mebs(0)
    # rhoBC = State.Strats.Deterministic.mebs(0)
    # rhoAC = State.Strats.Deterministic.mebs(0)
    qc = QuantumContext(random_variables=rvc, measurements=(A,B,C), states=(rhoAB, rhoBC, rhoAC), permutation=perm)
    pd = QuantumProbDist(qc)
    return pd

@profile
def go():
    symbolic_contexts = [
        [['C1'], ['A2', 'B2', 'C4']],
        [['C2'], ['A1', 'B2', 'C3']],
        [['C3'], ['A2', 'B1', 'C2']],
        [['C4'], ['A1', 'B1', 'C1']],
    ]
    inflation_rvc = RandomVariableCollection.new(names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4'], outcomes=[4,4,4,4,4,4,4,4])
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    defl_map = marginal_equality.get_delf_map(symbolic_contexts)
    pd = uniform_sample_qdistro(original_rvc)
    print(pd)
    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(pd, symbolic_contexts)
    res = positive_linear_solve.get_feasibility_cvx(A, b)
    pprint(res)

if __name__ == '__main__':
    go()
