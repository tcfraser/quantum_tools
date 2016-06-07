import numpy as np
from pprint import pprint
from . import marginal_equality
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities import utils
from ..utilities.profiler import profile
from ..statistics.variable import RandomVariableCollection
from . import positive_linear_solve
from ..examples import prob_dists

@profile
def go():
    # symbolic_contexts = [
    #     [['A2'], ['B2'], ['C2']],
    #     [['B2'], ['A2',   'C1']],
    #     [['C2'], ['A1',   'B2']],
    #     [['A2'], ['B1',   'C2']],
    #     [['A1',   'B1',   'C1']],
    # ]
    symbolic_contexts = [
        [['C1'], ['A2', 'B2', 'C4']],
        [['C2'], ['A1', 'B2', 'C3']],
        [['C3'], ['A2', 'B1', 'C2']],
        [['C4'], ['A1', 'B1', 'C1']],
    ]
    inflation_rvc = RandomVariableCollection.new(names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4'], outcomes=[4,4,4,4,4,4,4,4])
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    defl_map = marginal_equality.get_delf_map(symbolic_contexts)
    pd = prob_dists.fritz(original_rvc)

    print(pd)
    # print(pd.marginal(['A', 'B']))
    print(pd.condition({'C': 0}))
    # print(pd.condition({'C': 0}).correlation(['A', 'B']))
    # print(pd.condition({'C': 1}).correlation(['A', 'B']))
    # print(pd.condition({'C': 2}).correlation(['A', 'B']))
    # print(pd.condition({'C': 3}).correlation(['A', 'B']))
    CHSH = \
         - pd.condition({'C': 0}).correlation(['A', 'B']) \
         + pd.condition({'C': 1}).correlation(['A', 'B']) \
         + pd.condition({'C': 2}).correlation(['A', 'B']) \
         + pd.condition({'C': 3}).correlation(['A', 'B'])
    print(CHSH)

    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(pd, symbolic_contexts, defl_map)
    print(A.shape, b.shape)
    # print(len(A.nonzero()[0]))
    # print(len(b.nonzero()[0]))
    res = positive_linear_solve.get_feasibility_cvx(A, b)
    x_solution = res['x']
    Ax = A.dot(x_solution)
    bT = b[:,np.newaxis]
    # print(Ax.shape)
    # print(bT.shape)
    pprint(res)
    if not np.any(Ax - bT):
        print("Solution verified.")
    # print(res['x'])

if __name__ == '__main__':
    # go()

    import cProfile
    cProfile.run('go()', sort='time')

