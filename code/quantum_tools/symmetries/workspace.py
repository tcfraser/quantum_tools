import numpy as np
import sys, getopt
from collections import defaultdict
from itertools import permutations, product
from ..inflation import marginal_equality, marginal_equality_prune
from ..contexts.measurement import Measurement
from ..utilities import utils
from ..statistics.variable import RandomVariableCollection, RandomVariable
from ..statistics import variable_sort
from ..examples import prob_dists
from pprint import pprint
from scipy import sparse
from ..utilities import integer_map
from ..porta import marginal_problem_pipeline
from ..sets import *
from ..config import *
from operator import itemgetter
from ..examples.symbolic_contexts import *
from ..visualization.sparse_vis import plot_coo_matrix
import multiprocessing
from ..utilities import number_system
from functools import partial
from multiprocessing import Pool, cpu_count
# from ..esp import rayshooting

def perm_bin_op(a,b):
    return itemgetter(*a)(b)

def get_face_sets(rvc):
    face_map = defaultdict(list)
    for rv in rvc:
        face_map[rv.base_name].append(rv)
    return face_map

def permutations_of_face_sets(rvc):
    face_sets_dict = get_face_sets(rvc)
    posn_indices = [[rvc.names[rv.name] for rv in rvs] for rvs in face_sets_dict.values()]
    perm_group = [list(utils.flatten(p)) for p in permutations(posn_indices)]
    assert(utils.all_equal_length(posn_indices)), "posn_indices need to be all equal length."
    return perm_group

def get_permutations(n):
    return permutations(range(n))

def mtrx_rep_perm(perms):
    return [np.identity(len(p), dtype='int16')[p] for p in perms]

def group_avg_rep(reps):
    return sum(reps) / len(reps)

def get_orbits(actions, elems, indexof, num_elems=None):
    num_elems = num_elems if num_elems is not None else len(elems)
    seen = np.zeros(num_elems, dtype=bool)
    orbits = []
    for i, elem in enumerate(elems):
        if not seen[i]:
            orbit = []
            for action in actions:
                # print(elem)
                # print(action)
                image = action(elem)
                image_index = indexof(image)
                if not seen[image_index]:
                    orbit.append(image_index)
                    seen[image_index] = True
            orbits.append(orbit)
    return orbits

def find_actions(g_gen, bin_op):
    return _find_actions(g_gen, g_gen[0], set(), bin_op)

def _find_actions(g_gen, g_i, g_working, bin_op):
    if g_i not in g_working:
        g_working.add(g_i)
        for gen in g_gen:
            g_i = bin_op(gen, g_i)
            g_working.update(_find_actions(g_gen, g_i, g_working, bin_op))
    return g_working

def print_orbits(orbits):
    for i, orbit in enumerate(orbits):
        # print(orbit_indices)
        print("[Orbit {0}]".format(i+1))
        for elem in orbit:
            print(elem)

def get_sum_description(indices_structure, mtrx_constructor=None):
    if not callable(mtrx_constructor):
        raise ValueError("mtrx_constructor is not callable but needs to be.")
    indices = np.array(list(utils.flatten(indices_structure)))
    data = np.ones(len(indices))
    indptr = np.zeros(len(indices_structure) + 1)
    i = 0
    ptr = 0
    for index_set in indices_structure:
        i += 1
        ptr += len(index_set)
        indptr[i] = ptr
    # print(data)
    # print(len(data))
    # print(indices)
    # print(len(indices))
    # print(indptr)
    # print(len(indptr))
    mtrx = mtrx_constructor((data, indices, indptr))
    return mtrx

def get_col_sum_description(indices_structure):
    # print('col count', len(indices_structure))
    return get_sum_description(indices_structure, sparse.csc_matrix)

def get_row_sum_description(indices_structure):
    # print('row count', len(indices_structure))
    return get_sum_description(indices_structure, sparse.csr_matrix)

def generate_joint_outcomes_for_sc(rvc, symbolic_contexts):
    for sc_i in symbolic_contexts:
        rvs = utils.flatten(sc_i)
        sub_rvc = rvc.sub(rvs)
        prods = tuple(rv.outcomes if rv in sub_rvc else [-1] for rv in rvc)
        for i in product(*prods):
            yield i
        # print(sc_i)
        # print(prods)
        # sub_rvc_indices = rvc.names[sub_rvc.names.list]
        # for
        # print(sub_rvc_indices)

def num_sc_joint_outcome(rvc, symbolic_contexts):
    num = 0
    for sc_i in symbolic_contexts:
        rvs = utils.flatten(sc_i)
        sub_rvc = rvc.sub(rvs)
        num += len(sub_rvc.outcome_space)
    return num

def build_mblbt(rvc, symbolic_contexts):
    mblbt = number_system.MultiBaseLookupBTOptimized()
    # Non-marginal joint outcomes
    nmjob = tuple(rv.num_outcomes for rv in rvc)
    mblbt.register_base(nmjob) # fpbase

    # Marginal joint outcomes
    for sc_i in symbolic_contexts:
        rvs = utils.flatten(sc_i)
        sub_rvc = rvc.sub(rvs)
        sub_rvc_indices = rvc.names[sub_rvc.names.list]
        mblbt.register_base(tuple(rv.num_outcomes for rv in sub_rvc),
            base_indices=sub_rvc_indices, size=len(rvc), shift=True)

    return mblbt

def get_contraction(rvc, symbolic_contexts):
    jos = rvc.outcome_space
    num_jos = len(rvc.outcome_space)
    jos_sc = generate_joint_outcomes_for_sc(rvc, symbolic_contexts)
    num_jos_sc = num_sc_joint_outcome(rvc, symbolic_contexts)
    print(num_jos_sc, num_jos)
    mblbt = build_mblbt(rvc, symbolic_contexts)
    # TODO replace these
    if len(rvc) == 12:
        group_gen = ((3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8),
                     (0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7),
                     (8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7),
                     (0, 1, 2, 3, 5, 4, 7, 6, 10, 11, 8, 9))
        perms = find_actions(group_gen, perm_bin_op)
    # print(perms)
    # print(len(perms))
    # return (0,0)
    elif len(rvc) == 6:
        perms = permutations_of_face_sets(rvc)
    actions = [itemgetter(*perm) for perm in perms]
    # END TODO

    # Row summing
    row_orbits = get_orbits(actions, jos_sc, indexof=mblbt.get_val, num_elems=num_jos_sc)
    # row_orbit_indices = get_orbit_indices(row_orbits, jos_sc)
    row_sum = get_row_sum_description(row_orbits)
    print("Found {0} row_orbits.".format(len(row_orbits)))

    # # Col summing
    col_orbits = get_orbits(actions, jos, indexof=partial(mblbt.get_val, use_fpb=True), num_elems=num_jos)
    # col_orbit_indices = get_orbit_indices(col_orbits, jos)
    col_sum = get_col_sum_description(col_orbits)
    print("Found {0} col_orbits.".format(len(col_orbits)))

    # plot_coo_matrix(col_sum)
    # plot_coo_matrix(row_sum)

    # if ret_jo:
    #     return row_sum, jos_sc, col_sum, jos
    # else:
    return row_sum, col_sum

def profile_old():
    symbolic_contexts, outcomes = ABC_224_444
    rvc = RandomVariableCollection.new(
        names=marginal_equality.rv_names_from_sc(symbolic_contexts),
        outcomes=outcomes)
    row_sum, row_jo, col_sum, col_jo = get_contraction(rvc, symbolic_contexts, ret_jo=True)

def orbit_scale_test():
    elems = product(*((range(4)) for _ in range(12)))
    actions = []
    invariants = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    invariants = list(utils.chunks(list(range(12)), 4))
    for p in permutations([0,1,2]):
        actions.append(list(utils.flatten(itemgetter(*p)(invariants))))
    pprint(actions)
    f = lambda action, elem: itemgetter(*action)(elem)
    orbits = get_orbits(actions, elems, f=f)
    print(len(orbits))
    # print_orbits(orbits)
    # itemgetter()

def profile():
    get_contraction_elements(perform_pipeline=True)

def get_contraction_elements(sc, perform_pipeline=False):
    symbolic_contexts, outcomes = sc
    # symbolic_contexts, outcomes = ABC_222_222
    rvc = RandomVariableCollection.new(
        names=marginal_equality.rv_names_from_sc(symbolic_contexts),
        outcomes=outcomes)
    row_sum, col_sum = get_contraction(rvc, symbolic_contexts)
    A = marginal_equality.marginal_mtrx(rvc, symbolic_contexts)
    contracted_A = row_sum.dot(A.dot(col_sum))
    # print(row_sum.shape)
    # print(A.shape)
    # print(col_sum.shape)
    # print(contracted_A.shape)
    # print(contracted_A.nnz)

    if perform_pipeline:
        marginal_problem_pipeline.perform_pipeline(
            'contracted_big',
            contracted_A,
            optional_mtrxs={
                'row_sum': row_sum,
                'col_sum': col_sum,
                'contracted_A': contracted_A,
                'A': A,
        })
    return row_sum, A, col_sum, contracted_A

    # get_contraction(rvc, symbolic_contexts)
    # elems = rvc.outcome_space
    # actions = []
    # invariants = [[0,1],[2,3],[4,5]]
    # for p in permutations([0,1,2]):
    #     permutation_action = list(utils.flatten(itemgetter(*p)(invariants)))
    #     # print(permutation_action)
    #     actions.append(itemgetter(*permutation_action))

    # mblbt = number_system.MultiBaseLookupBTOptimized()
    # mblbt.register_base(
    #     tuple(rvc.outcome_space.get_input_base()),
    #     )
    # # pprint(actions)
    # # f = lambda action, elem: itemgetter(*action)(elem)
    # orbits = get_orbits(actions, elems, indexof=mblbt.get_val, num_elems=len(rvc.outcome_space))
    # print(len(orbits))
    # print_orbits(orbits)
    # itemgetter()

def test_gen_action():
    gen = [(0,1,2,3), (0,2,1,3), (1,0,3,2)]
    bin_op = lambda a,b: itemgetter(*a)(b)
    print(find_actions(gen, bin_op))

def scale_tests():
    a = (1,2,3,4,5,6,7,8)
    base = (128,64,32,16,8,4,2,1)
    perm = ((range(4)) for _ in range(12))
    # print(tuple(perm))
    hashes = {}
    # seen = set()
    # for i in range(4**12):
    #     seen.add(i)
    # for i in range(4**12):
    #     i in seen

    # return
    iterable = product(*perm)
    # hash_stash = np.zeros(4**12, dtype='int64')
    # hash_stash_dict = {}
    # with Pool(processes=cpu_count()-1) as pool:
    #     pooled_hash = pool.map(hash,iterable)
    # print(len(pooled_hash))
    # print(pooled_hash[0])
    for i, outcome in enumerate(iterable):
        hash_val = hash(outcome)
        hashes[hash_val] = i
        # hash_stash[i] = hash_val
        # hash_stash_dict[hash_val] = i

    # hash_keys =
    # for i in :
        # del hashes[i]

    # print(set(hash_stash))
    # for hash_val in hashes:
    #     assert(hash_val in hashes)
        # hash_stash_dict[hash_val]
    #     np.in1d
    # print(hash_stash.shape)
    # print(hash_stash[123456])
    # for i in :
    #     # print(i)
    #     hashes.add(hash(i))
        # sum(x*y for (x,y) in zip(a, base))

def test_number_system():
    # === Binary Tree ===
    ns = number_system.MultiBaseLookupBT()
    a = (-1,1,-1,2,-1,3,-1,4,-1,-1,-1,-1)
    ns.register_base(a,tuple(range(12)), shift=2)
    # hashes = {}
    for i in range(4**10):
        # hash(a)
        ns.get_val(a)
    # print(ns.get_val(a))

    # === Numpy lookup ===
    # ns = number_system.MultiBaseLookupNP(12)
    # a = (-1,1,-1,2,-1,3,-1,4,-1,-1,-1,-1)
    # a = np.asarray(a, dtype='int16')
    # ns.register_base(a,tuple(range(12)), shift=2)
    # for i in range(100000):
    #     ns.get_val(a)
    # print(ns.get_val(a))


if __name__ == '__main__':
    # PROFILE_MIXIN(profile)
    # PROFILE_MIXIN(scale_tests)
    # PROFILE_MIXIN(orbit_scale_test)
    PROFILE_MIXIN(profile)
    # PROFILE_MIXIN(test_gen_action)
    # PROFILE_MIXIN(test_number_system)
    # import timeit
    # print(timeit.timeit('[hash((1,2,3,4,5,6,7,8)) for i in range(4**12)]', number=1))
    # print(timeit.timeit('[[1,2,3,4,5,6,7,8][7,6,5,4,3,2,1,0] for i in range(4**12)]', number=1))
