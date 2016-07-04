from polytope import esp
import numpy as np
from scipy import sparse

def eliminate_marginals(A):
    r, c = A.shape
    R = np.zeros((2*r + c, r))
    C = np.zeros((2*r + c, c))
    b = np.zeros((2*r + c))

    # c positive
    for i in range(c):
        C[i,i] = -1

    # r equalities
    for i in range(r):
        R[c+i, i] = -1
        R[c+r+i, i] = +1

    A_array = A.toarray()
    C[c:c+r, 0:c] = A_array
    C[c+r:c+2*r, 0:c] = -A_array

    return R, C, b

def shoot(R,C,b):
    return esp.shoot(R,C,b)