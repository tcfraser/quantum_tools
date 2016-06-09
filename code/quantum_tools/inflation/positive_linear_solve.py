import numpy as np
import scipy
from scipy import io, sparse, optimize
# from scipy.sparse import linalg
# from scipy.optimize import linprog
import cvxopt as cvx
from ..utilities.profiler import profile
from ..utilities import integer_map
from .marginal_equality_prune import pre_process

# === Not supported on cluster machines ===
# def get_feasibility_scipy(A, b):
#     res = scipy.optimize.linprog(
#         c=-1*np.ones(A.shape[1]),
#         A_ub=None,
#         b_ub=None,
#         A_eq=A,
#         b_eq=b,
#         bounds=(0, None),
#         method='simplex',
#         callback=None,
#         options=None
#     )
#     if res.success:
#         assert(not np.any(np.dot(A, res.x) - b))
#     return res

def get_feasibility_cvx(A, b, prune=True):
    """
        Solves a pair of primal and dual LPs
            minimize    c'*x
            subject to  G*x + s = h
                        A*x = b
                        s >= 0
            maximize    -h'*z - b'*y
            subject to  G'*z + A'*y + c = 0
                        z >= 0.
        Input arguments.
            c is n x 1, G is m x n, h is m x 1, A is p x n, b is p x 1.  G and
            A must be dense or sparse 'd' matrices.  c, h and b are dense 'd'
            matrices with one column.  The default values for A and b are
            empty matrices with zero rows.
            solver is None, 'glpk' or 'mosek'.  The default solver (None)
            uses the cvxopt conelp() function.  The 'glpk' solver is the
            simplex LP solver from GLPK.  The 'mosek' solver is the LP solver
            from MOSEK.
    """
    print("A, b shapes", A.shape, b.shape)
    if prune:
        A, b = pre_process(A, b)
    print("A, b shapes", A.shape, b.shape)
    # if in coo format:
    if A.getformat() is not 'coo':
        A = A.tocoo()
    # print(A.row)
    # print(A.col)
    # print(A.data)
    # assert(False)

    x_size = A.shape[1]
    cvx_A = cvx.spmatrix(1.0, A.row, A.col, size=A.shape, tc='d')
    cvx_c = cvx.matrix(np.ones(x_size))
    cvx_G = cvx.spmatrix(-1.0, range(x_size), range(x_size), tc='d')
    cvx_h = cvx.matrix(np.zeros(x_size))
    cvx_b = cvx.matrix(b)
    res = cvx.solvers.lp(c=cvx_c,G=cvx_G,h=cvx_h,A=cvx_A,b=cvx_b,
        # solver=None,
        # solver='mosek',
        solver='glpk',
    )
    x_solution = res['x']
    Ax = A.dot(x_solution)
    bT = b[:,np.newaxis]
    verified = not np.any(Ax - bT) and np.all(np.array(x_solution) >= 0)
    res['verified'] = verified
    return res

def perform_tests():
    data = np.ones(9)
    i = np.array([0,2,1,0,2,1,1,0,0])
    j = np.array([0,1,2,3,4,5,6,7,8])
    A = sparse.coo_matrix((data, (i,j)))
    b = np.array([0.3, 0, 7])
    print(A.toarray())
    print(b)
    A_prune, b_prune = pre_process(A,b)
    print(A_prune.toarray())
    print(b_prune)

if __name__ == '__main__':
    perform_tests()
