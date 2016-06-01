import numpy as np
import scipy
from scipy import io, sparse, optimize
from scipy.sparse import linalg
from scipy.optimize import linprog
import cvxopt as cvx
from quantum_pd import gen_test_distribution, two_outcome_triangle
import global_config
from timing_profiler import timing

def get_LHS():
    LHS = scipy.io.mmread('../data/LHS_sparse.cua')
    return LHS

@timing
def check_feasibility():
    A = get_LHS().toarray()
    # LHS_inv = scipy.sparse.linalg.inv(LHS)
    for _ in range(1000):
        qpd = two_outcome_triangle()
        num_contexts = 5
        ravel_pd = qpd.ravel_support()
        b = np.tile(ravel_pd, num_contexts)[:,np.newaxis]
        b = np.ones(A.shape[0])
        # print(b)
        # print(qpd)
        # print(A.__class__)

        res = scipy.optimize.linprog(
            c=-1*np.ones(A.shape[1]),
            A_ub=None,
            b_ub=None,
            A_eq=A,
            b_eq=b,
            bounds=(0, None),
            method='simplex',
            callback=None,
            options=None
        )
        print(not np.any(np.dot(A, res.x) - b))
        print(res.success)

if __name__ == '__main__':
    check_feasibility()

# print(a)