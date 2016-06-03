"""
Contains python implementation of marginal setup.
"""
from operator import mul
from functools import reduce
import numpy as np
from ..utilities import utils
from ..statistics.variable import split_name
from ..statistics import variable_sort
from ..utilities.timing_profiler import timing
from ..config import *

def index_map_from_symbolic(symbolic_contexts):
    flat = utils.recurse(utils.flatten, 2, symbolic_contexts)
    unique = variable_sort.sort(utils.unique_everseen(flat))
    defl_map = dict((value, split_name(value)[0]) for key, value in enumerate(unique))
    return defl_map

def deflate(context, defl_map):
    return [[defl_map[rv] for rv in pre_inject] for pre_inject in context]

def context_marginals(pd, context, defl_map):
    defl_context = deflate(context, defl_map)
    marginals = map(pd.marginal, defl_context)
    product_marginals = reduce(mul, marginals)
    return np.array(list(product_marginals.canonical_ravel()))

def contexts_marginals(pd, contexts, defl_map):
    return np.hstack((context_marginals(pd, c, defl_map) for c in contexts))

@timing
def perform_tests():
    from ..examples.prob_dists import spekkens, demo_distro
    spd = spekkens()
    # spd = demo_distro()
    symbolic_contexts = [
        [['A2'], ['B2'], ['C2']],
        [['B2'], ['A2',   'C1']],
        [['C2'], ['A1',   'B2']],
        [['A2'], ['B1',   'C2']],
        [['A1',   'B1',   'C1']],
    ]
    # print(spd)
    defl_map = index_map_from_symbolic(symbolic_contexts)
    # print(infl_map)
    # print(defl_map)
    print(contexts_marginals(spd, symbolic_contexts, defl_map))

if __name__ == '__main__':
    perform_tests()