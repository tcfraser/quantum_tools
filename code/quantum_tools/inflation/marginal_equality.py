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
    defl_context = deflate(context, defl_map)
    # Map Reduce Parallelization can be done here.
    marginals = tuple(map(pd.marginal, defl_context))
    product_marginals = ProbDist.product_marginals(*marginals)
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

def contexts_marginals(pd, contexts):
    defl_map = get_delf_map(contexts)
    return np.hstack((context_marginals(pd, c, defl_map) for c in contexts))

def sparse_row(n):
    return sparse.coo_matrix(np.ones(n))

def multi_sparse_kron(*args):
    return reduce(lambda a, b: sparse.kron(a, b), args)

def marginal_mtrx_per_context(rvc, context):
    sub_rv_names = list(utils.unique_everseen(utils.flatten(context)))
    sub_rvc = rvc.sub(sub_rv_names)
    # print(sub_rvc)
    # sub_rv_names_indices = rvc.names[sub_rvc.names.list] # Needs to be off sub_rvc. It's sorted that way.
    kronecker_elems = tuple(sparse.identity(rv.num_outcomes) if rv in sub_rvc else sparse_row(rv.num_outcomes) for rv in rvc)
    return multi_sparse_kron(*kronecker_elems)
    # reduced_rvc_int_base = np.zeros(rvc.outcome_space.base_size)[:, np.newaxis]
    # # rvc_int_base[integer_map.comp_mask(sub_rv_names_indices, len(rvc.outcome_space))] = 0
    # # print(rvc_int_base)
    # # reduced_rvc_int_base = rvc_int_base[]
    # # np.zeros(rvc.outcome_space.base_size)[:, np.newaxis]
    # # print(sub_rvc.outcome_space.get_base().shape)
    # reduced_rvc_int_base[sub_rv_names_indices,:] = sub_rvc.outcome_space.get_base()


    # # reduced_rvc_int_base = np.array(reduced_rvc_int_base)[:, np.newaxis]
    # # print(reduced_rvc_int_base)
    # # print(sub_rv_names_indices)
    # # print(reduced_rvc_int_base)
    # # outcome_space = rvc.outcome_index_space
    # # sub_outcome_space = sub_rvc.outcome_index_space
    # # reduced_outcome_space = outcome_space[:, sub_rv_names_indices]
    # # marginal_mtrx_I = np.dot(reduced_outcome_space, sub_rvc.int_base)

    # non_zero_size = len(rvc.outcome_space)
    # # print(len(sub_rvc.outcome_space))
    # marginal_mtrx_J = np.arange(non_zero_size)
    # marginal_mtrx_I = np.squeeze(np.dot(rvc.outcome_space.cached_iter(), reduced_rvc_int_base))
    # # i = 0
    # # print(marginal_mtrx_I.shape)
    # # print(rvc.outcome_space.cached_iter().shape)
    # # # print(rvc.outcome_space)
    # # for outcome_idx in :
    # #     # print(outcome_idx)
    # #     # print(rvc.outcome_space._digits)
    # #     # print(sub_rv_names_indices)
    # #     # reduced_outcome_idx = outcome_idx[sub_rv_names_indices]
    #     # print(outcome_idx)
    #     # sub_row = sub_rvc.outcome_space.get_integer(outcome_idx, base=reduced_rvc_int_base)
    #     # print(sub_row)
    #     # print(sub_rv_names_indices)
    #     # print(reduced_rvc_int_base)
    #     # print(outcome_idx)
    #     # print(sub_row)
    #     # assert(i < 5)
    #     # marginal_mtrx_I[i] = sub_row
    #     # i += 1
    # # print(marginal_mtrx_I)
    # marginal_mtrx_data = np.ones(non_zero_size, dtype='int8')
    # marginal_mtrx_context = sparse.coo_matrix((marginal_mtrx_data, (marginal_mtrx_I, marginal_mtrx_J)), shape=(len(sub_rvc.outcome_space), len(rvc.outcome_space)), dtype='int8')
    # # print(marginal_mtrx_context.toarray())
    # return marginal_mtrx_context

def marginal_mtrx(rvc, contexts):
    return sparse.vstack((marginal_mtrx_per_context(rvc, context) for context in contexts))