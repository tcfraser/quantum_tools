import numpy as np
import sys, getopt
from ..inflation import marginal_equality, marginal_equality_prune
from ..contexts.measurement import Measurement
from ..utilities import utils
from ..statistics.variable import RandomVariableCollection
from ..examples import prob_dists
from ..visualization.sparse_vis import plot_coo_matrix

scABC_444__4 = [
    [['A1', 'B1', 'C1'], ['A4', 'B4', 'C4']],
    [['A1', 'B2', 'C3'], ['A4', 'B3', 'C2']],
    [['A2', 'B3', 'C1'], ['A3', 'B2', 'C4']],
    [['A2', 'B4', 'C3'], ['A3', 'B1', 'C2']],
    [['A1'], ['B3'], ['C4']],
    [['A1'], ['B4'], ['C2']],
    [['A2'], ['B1'], ['C4']],
    [['A2'], ['B2'], ['C2']],
    [['A3'], ['B3'], ['C3']],
    [['A3'], ['B4'], ['C1']],
    [['A4'], ['B1'], ['C3']],
    [['A4'], ['B2'], ['C1']],
]
scABC_444__4_outcomes = [4]*12

scABC_222__2 = [
    [['A2'], ['B2'], ['C2']],
    [['B2'], ['A2',   'C1']],
    [['C2'], ['A1',   'B2']],
    [['A2'], ['B1',   'C2']],
    [['A1',   'B1',   'C1']],
]
scABC_222__2_outcomes = [2]*6

scABC_224__4 = [
        [['C1'], ['A2', 'B2', 'C4']],
        [['C2'], ['A1', 'B2', 'C3']],
        [['C3'], ['A2', 'B1', 'C2']],
        [['C4'], ['A1', 'B1', 'C1']],
    ]
scABC_224__4_outcomes = [4]*(2 + 2 + 4)

def go(symbolic_contexts, outcomes):
    inflation_rvc = RandomVariableCollection.new(names=marginal_equality.rv_names_from_sc(symbolic_contexts), outcomes=outcomes)
    original_rvc = marginal_equality.deflate_rvc(inflation_rvc)
    print(inflation_rvc)
    print(original_rvc)
    fd = prob_dists.fritz(original_rvc)
    # inflation_rvc = RandomVariableCollection.new(names=marginal_equality.rv_names_from_sc(symbolic_contexts), outcomes=[4]*12)
    A = marginal_equality.marginal_mtrx(inflation_rvc, symbolic_contexts)
    b = marginal_equality.contexts_marginals(fd, symbolic_contexts)
    plot_coo_matrix(A)
    print("A.shape =", A.shape)
    print("A.nnz =", A.nnz)
    print("b.shape =", b.shape)
    A_pruned, b_pruned = marginal_equality_prune.pre_process(A,b)
    print("A_pruned.shape =", A_pruned.shape)
    print("A_pruned.nnz =", A_pruned.nnz)
    print("b_pruned.shape =", b_pruned.shape)

if __name__ == '__main__':
    # go()
    argv = sys.argv[1:]
    try:
       opts, args = getopt.getopt(argv,"s:p:",["size=","profile="])
    except getopt.GetoptError:
       print('-s <size> -p <profile>')
       sys.exit(2)
    for opt, arg in opts:
        if opt in ("-s", "--size"):
            size = arg
        if opt in ("-p", "--profile"):
            if arg in ['True', 't', 'true', 1]:
                should_profile = True
            if arg in ['False', 'f', 'false', 0]:
                should_profile = False

    if size == 'large':
        symbolic_contexts = scABC_444__4
        outcomes = scABC_444__4_outcomes
    elif size == 'small':
        symbolic_contexts = scABC_224__4
        outcomes = scABC_224__4_outcomes
    else:
        raise Exception("Needs size arg.")

    def go_wrapper():
        go(symbolic_contexts, outcomes)

    if should_profile:
        import cProfile

        cProfile.run('go_wrapper()', sort='time')
    else:
        go_wrapper()
