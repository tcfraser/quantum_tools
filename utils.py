import numpy as np
from scipy import linalg
from constants import *
from functools import reduce
import itertools

class Utils():
    @staticmethod
    def ei(x):
        """ Exponential notation for complex numbers """
        return np.exp(i*x)

    @staticmethod
    def tensor(*args):
        """ Implementation of tensor or kronecker product for tuple of matrices """
        return reduce(np.kron, args)

    @staticmethod
    def multiply(*args):
        """ Implementation of multi dot product between matrices """
        return reduce(np.dot, args)

    @staticmethod
    def is_hermitian(A):
        """ Checks if matrix A is hermitian """
        return np.array_equal(A, A.conj().T)

    @staticmethod
    def is_psd(A):
        """ Checks if matrix A is positive semi-definite """
        return np.all(linalg.eigvals(A) >= -mach_tol)

    @staticmethod
    def is_trace_one(A):
        """ Checks if matrix A has unitary trace or not """
        return is_close(np.trace(A), 1)

    @staticmethod
    def is_close(a,b):
        """ Checks if two numbers are close with respect to a machine tolerance defined above """
        return abs(a - b) < mach_tol

    @staticmethod
    def is_small(a):
        return Utils.is_close(a, 0)

    @staticmethod
    def gen_memory_slots(mem_loc):
        i = 0
        slots = []
        for m_size in mem_loc:
            slots.append(np.arange(i, i+m_size))
            i += m_size
        return slots

    @staticmethod
    def normalize(a):
        a /= linalg.norm(a)

    @staticmethod
    def ket_to_dm(ket):
        """ Converts ket vector into density matrix rho """
        return np.outer(ket, ket.conj())

    @staticmethod
    def entropy(x):
        if x != 0.0:
            return -x*np.log2(x)
        else:
            return 0.0

    @staticmethod
    def en_tuple(tup):
        if isinstance(tup, (tuple)):
            return tup
        elif isinstance(tup, (set, list)):
            return tuple(tup)
        else:
            return (tup,)

    @staticmethod
    def en_list(lst):
        if isinstance(lst, (list)):
            return lst
        elif isinstance(lst, (set, tuple)):
            return list(lst)
        else:
            return [lst]

    @staticmethod
    def en_set(lst):
        if isinstance(lst, (set)):
            return lst
        elif isinstance(lst, (list, tuple)):
            return set(lst)
        elif lst is None:
            return set()
        else:
            return set([lst])

    @staticmethod
    def get_permutation():
        perm = np.zeros((64,64), dtype=complex)
        buffer = np.empty((64,64), dtype=complex)
        for a in list(itertools.product(*[[0,1]]*6)):
            ket = Utils.tensor(*(qbs[a[i]] for i in (0,1,2,3,4,5)))
            bra = Utils.tensor(*(qbs[a[i]] for i in (1,2,3,4,5,0)))
            np.outer(ket, bra, buffer)
            perm += buffer
        return perm

    @staticmethod
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

Utils.v_entropy = np.vectorize(Utils.entropy)

def perform_tests():
    print(Utils.v_entropy(np.array([[0.0, 0.5], [0.5, 0.0]])))

if __name__ == '__main__':
    perform_tests()
    # @staticmethod
    # def norm_real_parameter(x):
    #     return np.cos(x)**2