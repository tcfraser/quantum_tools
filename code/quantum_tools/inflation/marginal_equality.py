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
from . import positive_linear_solve
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
    defl_context = deflate(context, defl_map)
    # Map Reduce Parallelization can be done here.
    marginals = map(pd.marginal, defl_context)
    product_marginals = reduce(ProbDist.product_marginals, marginals)
    # print(product_marginals)
    # print(list(product_marginals.canonical_ravel()))
    return np.array(list(product_marginals.ravel()))

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

def marginal_mtrx_per_context(rvc, context):
    sub_rv_names = list(utils.unique_everseen(utils.flatten(context)))
    sub_rvc = rvc.sub(sub_rv_names)
    # print(sub_rvc)
    sub_rv_names_indices = rvc.names[sub_rvc.names.list] # Needs to be off sub_rvc. It's sorted that way.
    reduced_rvc_int_base = [0]*rvc.outcome_space.base_size
    sub_rvc_int_base = sub_rvc.outcome_space.get_base()
    # print(reduced_rvc_int_base)
    for i, idx in enumerate(sub_rv_names_indices):
        reduced_rvc_int_base[idx] = sub_rvc_int_base[i]
    # print(sub_rv_names_indices)
    # print(reduced_rvc_int_base)
    # outcome_space = rvc.outcome_index_space
    # sub_outcome_space = sub_rvc.outcome_index_space
    # reduced_outcome_space = outcome_space[:, sub_rv_names_indices]
    # marginal_mtrx_I = np.dot(reduced_outcome_space, sub_rvc.int_base)

    non_zero_size = len(rvc.outcome_space)
    # print(len(sub_rvc.outcome_space))
    marginal_mtrx_J = np.arange(non_zero_size)
    marginal_mtrx_I = np.zeros(non_zero_size)
    i = 0
    # print(rvc.outcome_space)
    for outcome_idx in rvc.outcome_space:
        # print(outcome_idx)
        # print(rvc.outcome_space._digits)
        # print(sub_rv_names_indices)
        # reduced_outcome_idx = outcome_idx[sub_rv_names_indices]
        # print(outcome_idx)
        sub_row = sub_rvc.outcome_space.get_integer(outcome_idx, base=reduced_rvc_int_base)
        # print(sub_row)
        # print(sub_rv_names_indices)
        # print(reduced_rvc_int_base)
        # print(outcome_idx)
        # print(sub_row)
        # assert(i < 5)
        marginal_mtrx_I[i] = sub_row
        i += 1
    # print(marginal_mtrx_I)
    marginal_mtrx_data = np.ones(non_zero_size, dtype='int8')
    marginal_mtrx_context = sparse.coo_matrix((marginal_mtrx_data, (marginal_mtrx_I, marginal_mtrx_J)), shape=(len(sub_rvc.outcome_space), len(rvc.outcome_space)), dtype='int8')
    # print(marginal_mtrx_context.toarray())
    return marginal_mtrx_context

def marginal_mtrx(rvc, contexts):
    return sparse.vstack((marginal_mtrx_per_context(rvc, context) for context in contexts))