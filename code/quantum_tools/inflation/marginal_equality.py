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
from ..utilities.profiler import profile
from ..config import *
from . import positive_linear_solve

def get_delf_map(symbolic_contexts):
    flat = utils.recurse(utils.flatten, 2, symbolic_contexts)
    # unique = variable_sort.sort(utils.unique_everseen(flat)) # No need to sort here.
    unique = utils.unique_everseen(flat)
    defl_map = dict((value, split_name(value)[0]) for key, value in enumerate(unique))
    return defl_map

def deflate(context, defl_map):
    return [[defl_map[rv] for rv in pre_inject] for pre_inject in context]

def context_marginals(pd, context, defl_map):
    defl_context = deflate(context, defl_map)
    # Map Reduce Parallelization can be done here.
    marginals = map(pd.marginal, defl_context)
    product_marginals = reduce(mul, marginals)
    # print(product_marginals)
    # print(list(product_marginals.canonical_ravel()))
    return np.array(list(product_marginals.canonical_ravel()))

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

def contexts_marginals(pd, contexts, defl_map):
    return np.hstack((context_marginals(pd, c, defl_map) for c in contexts))

def outcome_index_space(rvc):
    return np.array([outcome_index for _, outcome_index, _ in rvc.outcome_space()])

def marginal_mtrx_per_context(rvc, context):
    sub_rv_names = list(utils.unique_everseen(utils.flatten(context)))
    sub_rvc = rvc.sub(sub_rv_names)
    sub_rv_names_indices = rvc.names[sub_rvc.names.list] # Needs to be off sub_rvc. It's sorted that way.
    outcome_space = outcome_index_space(rvc)
    sub_outcome_space = outcome_index_space(sub_rvc)
    reduced_outcome_space = outcome_space[:, sub_rv_names_indices]
    sub_outcome_reverse_map = np.arange(sub_outcome_space.shape[0]).reshape(sub_outcome_space[-1]+1)
    # marginal_mtrx = sparse.lil_matrix((len(sub_outcome_space), len(outcome_space)), dtype='int8')
    marginal_mtrx = np.zeros((len(sub_outcome_space), len(outcome_space)), dtype='int8')
    for i in range(len(reduced_outcome_space)):
        reduced_outcome = reduced_outcome_space[i]
        reduced_outcome_tuple = tuple(reduced_outcome)
        sub_outcome = sub_outcome_reverse_map[reduced_outcome_tuple]
        marginal_mtrx[sub_outcome, i] = 1
    marginal_mtrx = sparse.csc_matrix(marginal_mtrx, dtype='int8')
    return marginal_mtrx

def marginal_mtrx(rvc, contexts):
    return sparse.vstack((marginal_mtrx_per_context(rvc, context) for context in contexts))