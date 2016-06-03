import itertools
import numpy as np
from scipy import linalg
from functools import reduce
from operator import mul
from .constants import *

def ei(x):
    """ Exponential notation for complex numbers """
    return np.exp(i*x)

def tensor(*args):
    """ Implementation of tensor or kronecker product for tuple of matrices """
    return reduce(np.kron, args)

def multiply(*args):
    """ Implementation of multiple matrix multiplications between matrices """
    return reduce(mul, args)

def multidot(*args):
    """ Implementation of multiple dot product between matrices """
    return reduce(np.dot, args)

def is_hermitian(A):
    """ Checks if matrix A is hermitian """
    return np.array_equal(A, A.conj().T)

def is_psd(A):
    """ Checks if matrix A is positive semi-definite """
    return np.all(linalg.eigvals(A) >= -mach_tol)

def is_trace_one(A):
    """ Checks if matrix A has unitary trace or not """
    return is_close(np.trace(A), 1)

def is_close(a,b):
    """ Checks if two numbers are close with respect to a machine tolerance defined above """
    return abs(a - b) < mach_tol

def is_small(a):
    return is_close(a, 0)

def gen_memory_slots(mem_loc):
    i = 0
    slots = []
    for m_size in mem_loc:
        slots.append(np.arange(i, i+m_size))
        i += m_size
    return slots

def normalize(a):
    a /= linalg.norm(a)

def ket_to_dm(ket):
    """ Converts ket vector into density matrix rho """
    return np.outer(ket, ket.conj())

def __v_entropy(x):
    if x != 0.0:
        return -x*np.log2(x)
    else:
        return 0.0

def entropy(x):
    return np.sum(__v_entropy(x))

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def all_equal(iterator, to=None):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        if to is None:
            return all(first == rest for rest in iterator)
        else:
            return all(first == rest == to for rest in iterator)
    except StopIteration:
        return True

def en_tuple(tup):
    if isinstance(tup, (tuple)):
        return tup
    elif isinstance(tup, (set, list)):
        return tuple(tup)
    else:
        return (tup,)

def en_list(lst):
    if isinstance(lst, (list)):
        return lst
    elif isinstance(lst, (set, tuple)):
        return list(lst)
    else:
        return [lst]

def en_set(lst):
    if isinstance(lst, (set)):
        return lst
    elif isinstance(lst, (list, tuple)):
        return set(lst)
    elif lst is None:
        return set()
    else:
        return set([lst])

def get_permutation():
    perm = np.zeros((64,64), dtype=complex)
    for a in list(itertools.product(*[[0,1]]*6)):
        ket = tensor(*(qbs[a[i]] for i in (0,1,2,3,4,5)))
        bra = tensor(*(qbs[a[i]] for i in (1,2,3,4,5,0)))
        perm += np.outer(ket, bra)
    return perm

def largest_eig(M):
    size = M.shape[0]
    largest_eig = linalg.eigh(M, eigvals_only=True, eigvals=(size-1,size-1))[0]
    return largest_eig

def cholesky(t):
    # James, Kwiat, Munro, and White Physical Review A 64 052312
    # T = np.array([
    #     [      t[0]       ,       0         ,      0        ,   0  ],
    #     [  t[4] + i*t[5]  ,      t[1]       ,      0        ,   0  ],
    #     [ t[10] + i*t[11] ,  t[6] + i*t[7]  ,     t[2]      ,   0  ],
    #     [ t[14] + i*t[15] , t[12] + i*t[13] , t[8] + i*t[9] , t[3] ],
    # ])
    # assert(len(t) == size**2), "t is not the correct length. [len(t) = {0}, size = {1}]".format(len(t), size)
    size = int(len(t)**(1/2))
    indices = range(size)
    t_c = 0 # Parameter t counter
    T = np.zeros((size,size), dtype=complex)
    for row in indices:
        for col in indices:
            if (row == col): # diagonal
                T[row, col] = t[t_c]
                t_c += 1
            elif (row > col): # lower diagonal
                T[row, col] = t[t_c] + i * t[t_c + 1]
                t_c += 2
            elif (col > row): # upper diagonal
                pass
    Td = T.conj().T
    g = np.dot(Td, T)
    # assert(is_hermitian(g)), "g not hermitian!"
    # assert(is_psd(g)), "g not positive semi-definite!"
    return g

def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

def partial_identities(seed_desc):
    size = sum(seed_desc)
    As = []
    j = 0
    for i, seed in enumerate(seed_desc):
        A = np.zeros((size, size))
        for _ in range(seed):
            A[j, j] = 1
            j += 1
        As.append(A)
    return As

def param_GL_C(t):
    assert(is_square(len(t)/2))
    size = int((len(t)/2)**(1/2))
    t = np.asarray(t)
    GL_RR = np.reshape(t, (2, size, size))
    GL_C = GL_RR[0] + i * GL_RR[1]
    return GL_C

def get_meas_on_bloch_sphere(theta,phi):
    psi = np.cos(theta) * qb0 + np.sin(theta) * ei(phi) * qb1
    dm = ket_to_dm(psi)
    return dm

def get_orthogonal_pair(t):
    theta = t[0]
    phi = t[1]
    psi_1 = np.cos(theta) * qb0 - np.sin(theta) * ei(phi) * qb1
    psi_2 = np.sin(theta) * qb0 + np.cos(theta) * ei(phi) * qb1
    dm_1 = ket_to_dm(psi_1)
    dm_2 = ket_to_dm(psi_2)
    # print(dm_1)
    # print(dm_2)
    # print(dm_1 + dm_2)
    return dm_1, dm_2

__v_entropy = np.vectorize(__v_entropy)

def perform_tests():
    print(np.__version__)
    # return
    print(__v_entropy(np.array([[0.0, 0.5], [0.5, 0.0]])))

if __name__ == '__main__':
    perform_tests()
    # def norm_real_parameter(x):
    #     return np.cos(x)**2