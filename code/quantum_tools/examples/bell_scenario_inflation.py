import numpy as np
from pprint import pprint
from ..inflation import marginal_equality
from ..contexts.measurement import Measurement
from ..contexts.state import State
from ..contexts.quantum_context import QuantumContext, QuantumProbDist
from ..utilities import utils
from ..utilities.profiler import profile
from ..statistics.variable import RandomVariableCollection
from ..examples import prob_dists
from ..statistics.probability import ProbDist
from ..visualization import sparse_vis
from ..porta import marginal_problem_pipeline, porta_tokenizer
from ..config import *
from ..examples.symbolic_contexts import *

def set_up():
    symbolic_contexts, outcomes = ABXY_2222_2222
    inflation_rvc = RandomVariableCollection.new(names=marginal_equality.rv_names_from_sc(symbolic_contexts), outcomes=outcomes)
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    # print(original_rvc.sub('A'))
    # print(str(original_rvc))
    # return original_rvc
    # print(original_rvc)
    pd = prob_dists.tsirelson(original_rvc)
    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(pd, symbolic_contexts)
    return A, b, symbolic_contexts, inflation_rvc, original_rvc, pd

def linear_feasibility():
    from ..inflation import positive_linear_solve
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
    # set_up()
    # import cProfile
    # cProfile.run('fmel()', sort='time')