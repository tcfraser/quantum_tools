import numpy as np
from scipy import linalg
from constants import *

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
    def norm_real_parameter(x):
        return np.cos(x)**2