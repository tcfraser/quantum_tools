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
from ..statistics.probability import ProbDist
from ..visualization import sparse_vis
from ..porta import marginal_problem_pipeline, porta_tokenizer
from ..config import *

def set_up():
    symbolic_contexts = [
        [['X1', 'Y1'], ['A2', 'B2', 'X2', 'Y2']],
        [['X1', 'Y2'], ['A2', 'B1', 'X2', 'Y1']],
        [['X2', 'Y1'], ['A1', 'B2', 'X1', 'Y2']],
        [['X2', 'Y2'], ['A1', 'B1', 'X1', 'Y1']],
    ]
    inflation_rvc = RandomVariableCollection.new(names=['A1', 'A2', 'B1', 'B2', 'X1', 'X2', 'Y1', 'Y2'], outcomes=[2]*8)
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    pd = prob_dists.tsirelson(original_rvc)
    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(pd, symbolic_contexts)
    return A, b, symbolic_contexts, inflation_rvc, original_rvc, pd

def linear_feasibility():
    A, b, symbolic_contexts, inflation_rvc, original_rvc, pd = set_up()
    res = positive_linear_solve.get_feasibility_cvx(A, b)
    print(res)
    # np.savetxt(OUTPUT_DIR + "x_bell_scenario_256_256.csv", np.array(res['x']))

def fmel():
    A, b, symbolic_contexts, inflation_rvc, original_rvc, pd = set_up()
    marginal_problem_pipeline.perform_pipeline('bell_scenario', A)

if __name__ == '__main__':
    # linear_feasibility()
    fmel()
    # import cProfile
    # cProfile.run('fmel()', sort='time')