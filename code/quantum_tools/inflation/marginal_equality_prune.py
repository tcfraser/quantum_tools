import numpy as np
import scipy
from scipy import sparse
# from scipy.sparse import linalg
# from scipy.optimize import linprog
from ..utilities import integer_map

def pre_process(A, b):
    if A.format != 'csr':
        A_csr = A.tocsr()
    else:
        A_csr = A
    A_csc = A_csr.tocsc()
    zero_b = np.where(b == 0.0)[0]
    A_row_zero_b = A_csr[zero_b, :]
    col_to_delete = A_row_zero_b.indices # Over counts
    rows_to_delete = zero_b
    rows_to_keep = integer_map.comp_mask(rows_to_delete, A.shape[0])
    col_to_keep = integer_map.comp_mask(col_to_delete, A.shape[1])

    A_col_removed = A_csc[:,col_to_keep]
    A_row_removed = A_col_removed[rows_to_keep,:]

    A_pruned = A_row_removed

    b_pruned = np.delete(b, zero_b, axis=0)

    return A_pruned, b_pruned
