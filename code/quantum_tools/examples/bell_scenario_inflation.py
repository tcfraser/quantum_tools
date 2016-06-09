import numpy as np
from pprint import pprint
from ..inflation import marginal_equality
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities import utils
from ..utilities.profiler import profile
from ..statistics.variable import RandomVariableCollection
from ..inflation import positive_linear_solve
from ..examples import prob_dists
from ..visualization import sparse_vis
from ..config import *

@profile
def go():
    symbolic_contexts = [
        [['X1', 'Y1'], ['A2', 'B2', 'X2', 'Y2']],
        [['X1', 'Y2'], ['A2', 'B1', 'X2', 'Y1']],
        [['X2', 'Y1'], ['A1', 'B2', 'X1', 'Y2']],
        [['X2', 'Y2'], ['A1', 'B1', 'X1', 'Y1']],
    ]
    inflation_rvc = RandomVariableCollection.new(names=['A1', 'A2', 'B1', 'B2', 'X1', 'X2', 'Y1', 'Y2'], outcomes=[2]*8)
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    pd = prob_dists.tsirelson(original_rvc)
    print(pd)
    print(pd.marginal(['X', 'Y']))

    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(pd, symbolic_contexts)
    # for i in range(64):
    #     print(b[i])
    res = positive_linear_solve.get_feasibility_cvx(A, b)
    pprint(res)
    # sparse_vis.plot_coo_matrix(A)

    np.savetxt(OUTPUT_DIR + "x_bell_scenario_256_256.csv", np.array(res['x']))
    # print(np.array(b))
    # print(np.array(res['x']))

if __name__ == '__main__':
    go()

    # import cProfile
    # cProfile.run('go()', sort='time')