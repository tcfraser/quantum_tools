import numpy as np
import sys, getopt
from collections import defaultdict
from itertools import permutations
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
from operator import itemgetter
from ..examples.symbolic_context import *
from ..visualization.sparse_vis import plot_coo_matrix

class IndivOutcome():
    # Hashable

    def __init__(self, rv, outcome):
        self.rv = rv
        self.outcome = outcome
        self.__hash = hash(self.rv.name + str(self.outcome))
        assert(outcome in rv.outcomes), "Outcome {0} not possible.".format(outcome)

    def __repr__(self):
        _repr = "{name} = {0}".format(self.outcome, name=self.rv.name)
        return _repr

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(self.__hash)

class JointOutcome(SortedFrozenSet):

    @classmethod
    def __sort__(cls, io):
        return variable_sort.alphanum_key(str(io))

    def __post_init__(self):
        # Sort is ensured by SortedFrozenSet
        self.names = [io.rv.name for io in self]
        self.outcomes = [io.outcome for io in self]
        self._sort_key = ''.join(self.names) + ''.join((str(i) for i in self.outcomes))
        assert(len(self.names) == len(set(self.names))), "There is an overlap in random variables: {0}.".format(names)

    def __repr__(self):
        _repr = ', '.join(str(io) for io in self)
        return _repr

    def __str__(self):
        return repr(self)

WrapMethods(JointOutcome)

class JointOutcomes(SortedFrozenSet):

    @classmethod
    def __sort__(cls, jo):
        return jo._sort_key

    def __repr__(self):
        _repr = '\n'.join(str(jo) for jo in self)
        return _repr

    def __str__(self):
        return repr(self)

WrapMethods(JointOutcomes)

def change_io_name(io, new_name):
    return IndivOutcome(RandomVariable(new_name, io.rv.outcomes), io.outcome)

def flip_name(name_map, jo):
    io_list = [change_io_name(io, name_map[io.rv.name]) for io in jo]
    return JointOutcome(io_list)

def generate_joint_outcomes(rvc):
    rvs = list(rvc)
    jos = []
    for outcomes in rvc.outcome_space:
        ios = []
        for rv, outcome in zip(rvs, outcomes):
            ios.append(IndivOutcome(rv, outcome))
        jos.append(JointOutcome(ios))
    jos = JointOutcomes(jos)
    return jos

def generate_joint_outcomes_for_sc(rvc, sc):
    jos = []
    for context in sc:
        subrvc = rvc.sub(utils.flatten(context))
        subrvs = list(subrvc)
        for outcomes in subrvc.outcome_space:
            ios = []
            for rv, outcome in zip(subrvs, outcomes):
                ios.append(IndivOutcome(rv, outcome))
            jos.append(JointOutcome(ios))
    jos = JointOutcomes(jos)
    return jos

def get_face_sets(rvc):
    face_map = defaultdict(list)
    for rv in rvc:
        face_map[rv.base_name].append(rv.name)
    return face_map

def permutations_of_face_sets(rvc):
    face_sets_dict = get_face_sets(rvc)
    face_values = face_sets_dict.values()
    perm_group = [list(utils.flatten(p)) for p in permutations(face_values)]
    perm_maps = [dict(zip(perm_group[0], p)) for p in perm_group]
    # print()
    # pprint(perm_maps)
    # pprint(list(permutations(face_sets_dict.values())))
    # posn_indices = [[posn_of(rv.name) for rv in rvs] for rvs in face_sets_dict.values()]
    # # assert(utils.all_equal_length(posn_indices)), "posn_indices need to be all equal length."
    # base_names = face_sets_dict.keys()
    return perm_maps

def get_permutations(n):
    return permutations(range(n))

def mtrx_rep_perm(perms):
    return [np.identity(len(p), dtype='int16')[p] for p in perms]

def group_avg_rep(reps):
    return sum(reps) / len(reps)

def get_orbits(actions, elems, f=None, hasher=hash):
    f = f if f is not None else np.dot
    occupied = set()
    orbits = []
    for elem in elems:
        hashed_elem = hasher(elem)
        if hashed_elem not in occupied:
            orbit = set()
            for action in actions:
                res = f(action, elem)
                res_hash = hasher(res)
                if res_hash not in occupied:
                    orbit.add(res)
                    occupied.add(res_hash)
            orbit = list(orbit)
            orbits.append(orbit)
    return orbits

def get_orbit_indices(orbits, elems):
    return [[elems.index(o_elem) for o_elem in orbit] for orbit in orbits]

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
    jos = generate_joint_outcomes(rvc)
    jos_sc = generate_joint_outcomes_for_sc(rvc, symbolic_contexts)
    perms = permutations_of_face_sets(rvc)

    # Row summing
    row_orbits = get_orbits(perms, jos_sc, f=flip_name)
    row_orbit_indices = get_orbit_indices(row_orbits, jos_sc)
    row_sum = get_row_sum_description(row_orbit_indices)

    # Col summing
    col_orbits = get_orbits(perms, jos, f=flip_name)
    col_orbit_indices = get_orbit_indices(col_orbits, jos)
    col_sum = get_col_sum_description(col_orbit_indices)

    # plot_coo_matrix(col_sum)
    # plot_coo_matrix(row_sum)

    if ret_jo:
        return row_sum, jos_sc, col_sum, jos
    else:
        return row_sum, col_sum

symbolic_contexts = scABC_222__2
outcomes = scABC_222__2_outcomes
rvc = RandomVariableCollection.new(
    names=marginal_equality.rv_names_from_sc(symbolic_contexts),
    outcomes=outcomes)
# original_rvc = marginal_equality.deflate_rvc(inflation_rvc)

row_sum, row_jo, col_sum, col_jo = get_contraction(rvc, symbolic_contexts, ret_jo=True)
A = marginal_equality.marginal_mtrx(rvc, symbolic_contexts)
contracted_A = row_sum.dot(A.dot(col_sum))
print(row_sum.shape)
print(A.shape)
print(col_sum.shape)
print(contracted_A.shape)
pf = marginal_problem_pipeline.perform_pipeline("A1A2B1B2C1C2_222222_test", contracted_A)
# plot_coo_matrix(contracted_A)

# print(jos_sc)
# print(jos)
# jos = generate_joint_outcomes(inflation_rvc.sub(utils.flatten(symbolic_contexts[0])))

# for i in inflation_rvc.outcome_space:
#     print(i)

# print(inflation_rvc)
# print(original_rvc)
# outcome_list = inflation_rvc.outcome_space.cached_iter()
# outcome_base = inflation_rvc.outcome_space.get_base()
# outcome_hasher = inflation_rvc.outcome_space.get_integer
# # ex_cop = cop[45]

# rep_perms = mtrx_rep_perm(perms)
# avg_p = group_avg_rep(rep_perms)
# print(avg_p)
# print(utils.multidot(avg_p.T, avg_p))
# print(ex_cop)
# print(perms[3])
# print(ex_cop[perms[3]])
# print(utils.multidot(ex_cop, rep_perms[3]))
# print(outcome_list)
# print(utils.multidot(outcome_list, avg_p.T))
# print(utils.multidot(avg_p.T, np.diag(outcome_base)))
# projective_outcomes = utils.multidot(outcome_list, avg_p.T)
# orbits = get_orbits(rep_perms, outcome_list, hasher=outcome_hasher)
# orbits = get_orbits(perms, jos, f=flip_name)
# print(row_sum)
# plot_coo_matrix(row_sum)
# print_orbits(orbits)
# print(orbit_indices)


# for orbit in orbits:
#     # print(orbit_indices)
#     print("---")
#     for i in orbit:
#         print(jos_sc.index(i))
# print()

# io1 = IndivOutcome(RandomVariable('A1', 2), 0)
# io2 = IndivOutcome(RandomVariable('A2', 2), 1)
# print(hash([io1, io2]))
# print(io1)
# print(io2)
# a = JointOutcome([io1,io2])
# print(a)
# b = flip_base_name(a, {'A':'B', 'B':'A'})
# print(b)
# b = JointOutcome([io2,io1])
# c = a.union(b)
# print(c)
# print(c)

# for p in perms:
#     print(p)
# for p in rep_perms:
#     print(p)
#     print(p.T)
#     print(utils.multidot(p, p.T))
    # print(inflation_rvc[p])
# for i in get_permutations(4):
#     print(i)

# for sc in symbolic_contexts:
#     fsc = utils.flatten(sc)
#     rvc = inflation_rvc.sub(fsc)
#     print(rvc)

# print(inflation_rvc.outcome_space.get_integer([0,1,1,0,1,0]))