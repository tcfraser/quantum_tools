"""
Contains python implementation of marginal setup.
"""
from operator import mul
from functools import reduce
from scipy import sparse
import numpy as np
from ..utilities import utils
from ..statistics.variable import split_name, RandomVariableCollection, RandomVariable
from ..statistics import variable_sort
from ..statistics.probability import ProbDist
from ..utilities.profiler import profile
from ..config import *
from ..utilities import integer_map
import time

def get_delf_map(symbolic_contexts):
    unique = rv_names_from_sc(symbolic_contexts)
    defl_map = dict((value, split_name(value)[0]) for key, value in enumerate(unique))
    return defl_map

def rv_names_from_sc(symbolic_contexts):
    # unique = variable_sort.sort(utils.unique_everseen(flat)) # No need to sort here.
    flat = list(utils.recurse(utils.flatten, 2, symbolic_contexts))
    names = list(utils.unique_everseen(flat))
    return names

def deflate(context, defl_map):
    return [[defl_map[rv] for rv in pre_inject] for pre_inject in context]

def context_marginals(pd, context, defl_map):
    infl_rv_names = list(utils.flatten(context))
    infl_rv_names_arg_sort = variable_sort.argsort(infl_rv_names)
    defl_context = deflate(context, defl_map)
    marginals = tuple(map(pd.marginal, defl_context))
    product_marginals = ProbDist.product_marginals(*marginals, transpose_sort=False)
    product_marginals = np.transpose(product_marginals, axes=infl_rv_names_arg_sort)
    return np.asarray(product_marginals.ravel())

def deflate_rvc(rvc):
    base_names = utils.unique_everseen(rv.base_name for rv in rvc)
    deflated_rvs = []
    for base_name in base_names:
        base_name_sub_rvc = rvc.sub_base_name(base_name)
        outcomes_in_sub = list(rv.outcomes for rv in base_name_sub_rvc)
        assert(utils.all_equal(outcomes_in_sub)), "Outcomes of random variables not equal. {0}".format(str(outcomes_in_sub))
        outcome_for_base_name_set = outcomes_in_sub[0] # they're all equal
        deflated_rvs.append(RandomVariable(base_name, outcome_for_base_name_set))
    outcomes = utils.unique_everseen(rv.base_name for rv in rvc)
    return RandomVariableCollection(deflated_rvs)

def contexts_marginals(pd, contexts):
    defl_map = get_delf_map(contexts)
    return np.hstack((context_marginals(pd, c, defl_map) for c in contexts))

def sparse_row(n):
    return sparse.coo_matrix(np.ones(n))

def multi_sparse_kron(*args):
    return reduce(lambda a, b: sparse.kron(a, b, format='coo'), args)

def marginal_mtrx_per_context(rvc, context):
    sub_rv_names = list(utils.unique_everseen(utils.flatten(context)))
    sub_rvc = rvc.sub(sub_rv_names)
    kronecker_elems = tuple(sparse.identity(rv.num_outcomes) if rv in sub_rvc else sparse_row(rv.num_outcomes) for rv in rvc)
    return multi_sparse_kron(*kronecker_elems)

def marginal_mtrx(rvc, contexts):
    return sparse.vstack((marginal_mtrx_per_context(rvc, context) for context in contexts))