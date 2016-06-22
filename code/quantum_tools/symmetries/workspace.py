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
from multiprocessing import Pool, cpu_count


def get_face_sets(rvc):
    face_map = defaultdict(list)
    for rv in rvc:
        face_map[rv.base_name].append(rv)
    return face_map

def permutations_of_face_sets(rvc):
    face_sets_dict = get_face_sets(rvc)
    posn_indices = [[posn_of(rv.name) for rv in rvs] for rvs in face_sets_dict.values()]
    perm_group = [list(utils.flatten(p)) for p in permutations(posn_indices)]
    assert(utils.all_equal_length(posn_indices)), "posn_indices need to be all equal length."
    return perm_maps

def get_permutations(n):
    return permutations(range(n))

def mtrx_rep_perm(perms):
    return [np.identity(len(p), dtype='int16')[p] for p in perms]

def group_avg_rep(reps):
    return sum(reps) / len(reps)

def get_orbits(actions, elems, f=None, hasher=hash):
    f = f if f is not None else np.dot
    hashed_elems = {}
    for i, elem in enumerate(elems):
        hashed_elem = hasher(elem)
        hashed_elems[elem] = i
    orbits = []
    for elem in elems:
        # print(elem)
        hashed_elem = hasher(elem)
        if hashed_elem not in seen:
            orbit = []
            for action in actions:
                res = f(action, elem)
                res_hash = hasher(res)
                if res_hash not in seen:
                    orbit.append(res)
                    seen[res_hash]
            orbit = orbit
            orbits.append(orbit)
    return orbits

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

def get_contraction(rvc, symbolic_contexts, ret_jo=False):
    jos = rvc.generate_joint_outcomes(rvc)
    jos_sc = generate_joint_outcomes_for_sc(rvc, symbolic_contexts)
    perms = permutations_of_face_sets(rvc)

    # Row summing
    row_orbits = get_orbits(perms, jos_sc, f=flip_name)
    row_orbit_indices = get_orbit_indices(row_orbits, jos_sc)
    row_sum = get_row_sum_description(row_orbit_indices)
    print("Found {0} row_orbits.".format(len(row_orbits)))

    # Col summing
    col_orbits = get_orbits(perms, jos, f=flip_name)
    col_orbit_indices = get_orbit_indices(col_orbits, jos)
    col_sum = get_col_sum_description(col_orbit_indices)
    print("Found {0} col_orbits.".format(len(col_orbits)))

    # plot_coo_matrix(col_sum)
    # plot_coo_matrix(row_sum)

    if ret_jo:
        return row_sum, jos_sc, col_sum, jos
    else:
        return row_sum, col_sum

def profile():
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
    hash_stash = np.zeros(4**12, dtype='int64')
    # hash_stash_dict = {}
    # with Pool(processes=cpu_count()-1) as pool:
    #     pooled_hash = pool.map(hash,iterable)
    # print(len(pooled_hash))
    # print(pooled_hash[0])
    for i, outcome in enumerate(iterable):
        hash_val = hash(outcome)
        hashes[hash_val] = i
        hash_stash[i] = hash_val
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
    ns = number_system.MultiBaseLookup()
    a = (-1,1,-1,2,-1,3,-1,4,-1,-1,-1,-1)
    ns.register_base(a,tuple(range(12)))
    for i in range(1000000):
        ns.get_val(a)


if __name__ == '__main__':
    # PROFILE_MIXIN(profile)
    # PROFILE_MIXIN(scale_tests)
    # PROFILE_MIXIN(orbit_scale_test)
    PROFILE_MIXIN(test_number_system)
    # import timeit
    # print(timeit.timeit('[hash((1,2,3,4,5,6,7,8)) for i in range(4**12)]', number=1))
    # print(timeit.timeit('[[1,2,3,4,5,6,7,8][7,6,5,4,3,2,1,0] for i in range(4**12)]', number=1))
