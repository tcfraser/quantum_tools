import numpy as np
import sys, getopt
from ..inflation import marginal_equality, marginal_equality_prune
from ..contexts.measurement import Measurement
from ..utilities import utils
from ..statistics.variable import RandomVariableCollection
from ..examples import prob_dists
from ..visualization.sparse_vis import plot_coo_matrix
from ..examples.symbolic_contexts import *

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
            if arg in ['Large', 'large']:
                symbolic_contexts = scABC_444__4
                outcomes = scABC_444__4_outcomes
            else:
                symbolic_contexts = scABC_224__4
                outcomes = scABC_224__4_outcomes
        if opt in ("-p", "--profile"):
            if arg in ['True', 't', 'true', 1]:
                should_profile = True
            if arg in ['False', 'f', 'false', 0]:
                should_profile = False

    if should_profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        go(symbolic_contexts, outcomes)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.strip_dirs()
        ps.print_stats(.2)
    else:
        go(symbolic_contexts, outcomes)
