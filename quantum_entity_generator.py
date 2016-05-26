import itertools
import numpy as np
from scipy import linalg
from functools import reduce
from utils import *
from timing_profiler import timing

# === Configure ===
np.set_printoptions(precision=3, linewidth=120, suppress=True)

# === Constants ===
i = 1j
mach_tol = 1e-12
mach_eps = 1e-20
I2 = np.eye(2)
I4 = np.eye(4)
qb0 = np.array([[1], [0]], dtype=complex) # Spin down (ground state)
qb1 = np.array([[0], [1]], dtype=complex) # Spin up (excited state)
qbs = np.array([qb0, qb1])
qb00 = np.kron(qb0, qb0)
qb01 = np.kron(qb0, qb1)
qb10 = np.kron(qb1, qb0)
qb11 = np.kron(qb1, qb1)
sigx = np.array([[0,1],[1,0]])
sigy = np.array([[0,-i],[i,0]])
sigz = np.array([[1,0],[0,-1]])
tqbs = np.array([qb00, qb01, qb10, qb11])

# === Entity Generators ===

def ket_to_dm(ket):
    """ Converts ket vector into density matrix rho """
    return np.outer(ket, ket.conj())

def get_two_qubit_state(s):
    assert (len(s) == 7), "7 Parameters are needed to form an arbitrary two qubit state."
    state = qb00*s[0] + qb01*s[1]*ei(s[2]) + qb10*s[3]*ei(s[4]) + qb11*s[5]*ei(s[6])
    normalize(state)
    return state

def get_perumation():
    perm = np.zeros((64,64), dtype=complex)
    buffer = np.empty((64,64), dtype=complex)
    for a in list(itertools.product(*[[0,1]]*6)):
        ket = tensor(*(qbs[a[i]] for i in (0,1,2,3,4,5)))
        bra = tensor(*(qbs[a[i]] for i in (1,2,3,4,5,0)))
        np.outer(ket, bra, buffer)
        perm += buffer
    return perm

def get_maximally_entangled_bell_state(n=0):
    n = n % 4
    norm = 1/np.sqrt(2)
    if n == 0:
        psi = norm * (qb00 + qb11)
    elif n == 1:
        psi = norm * (qb00 - qb11)
    elif n == 2:
        psi = norm * (qb01 + qb10)
    elif n == 3:
        psi = norm * (qb01 - qb10)
    rho = ket_to_dm(psi)
    return rho

def get_psd_herm_ntr(t, size):
    g = cholesky(t, size)
    g /= (np.trace(g) + mach_eps)
    # assert(is_trace_one(groundn)), "Trace of g is not 1.0! Difference: {0}".format(np.trace(g) - 1)
    return g

def get_psd_herm_neig(t, size):
    g = cholesky(t, size)
    largest_eig_val = linalg.eigh(g, eigvals_only=True, eigvals=(size-1,size-1))[0]
    if (largest_eig_val > 1.0):
            g /= (largest_eig_val + mach_eps)
    return g

def get_correl_meas(M_y):
    In = np.eye(M_y.shape[0])
    M_n = In - M_y
    dM = M_y - M_n
    return dM

# @timing
def pvms(t, size):
    g = cholesky(t, size)
    eigen_values, eigen_vectors = linalg.eigh(g)
    density_matrices = [ket_to_dm(eigen_vectors[:,i]) for i in range(size)]
    return density_matrices
    # return sum(density_matrices)

def cholesky(t, size):
    # James, Kwiat, Munro, and White Physical Review A 64 052312
    # T = np.array([
    #     [      t[0]       ,       0         ,      0        ,   0  ],
    #     [  t[4] + i*t[5]  ,      t[1]       ,      0        ,   0  ],
    #     [ t[10] + i*t[11] ,  t[6] + i*t[7]  ,     t[2]      ,   0  ],
    #     [ t[14] + i*t[15] , t[12] + i*t[13] , t[8] + i*t[9] , t[3] ],
    # ])
    assert(len(t) == size**2), "t is not the correct length. [len(t) = {0}, size = {1}]".format(len(t), size)
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

def get_projective_meas(weight, param):
    # weight = min(max(weight, 0), 1)
    assert (len(param) == 7), "Projective state needs 7 parameters."
    # assert (0 <= weight <= 1), "Weight {w} is not normalized.".format(w=weight)
    state = get_two_qubit_state(param)
    M_y = norm_real_parameter(weight) * ket_to_dm(state)
    M_n = I4 - M_y
    dM = M_y - M_n
    return dM

def get_meas_on_bloch_sphere(theta,phi):
    u_1 = np.cos(phi) * np.sin(theta)
    u_2 = np.sin(phi) * np.sin(theta)
    u_3 = np.cos(theta)
    sig_u = u_1 * sigx + u_2 * sigy + u_3 * sigz
    return sig_u

def get_two_qubit_diagonal_state(q):
    return np.cos(np.pi * q) * qb00 + np.sin(np.pi * q) * qb11

def get_tqds_dm(q):
    return ket_to_dm(get_two_qubit_diagonal_state(q))

def __tests__():
    print(pvms(np.random.random(16), 4))
    # print(get_maximally_entangled_bell_state(0))
    # print(get_maximally_entangled_bell_state(1))
    # print(get_maximally_entangled_bell_state(2))
    # print(get_maximally_entangled_bell_state(3))
    # print(get_maximally_entangled_bell_state(4))

    # param = np.random.normal(scale=1.0, size=16)
    # T_1 = get_psd_herm(param, 4, unitary_trace=True)
    # # print(T_1)
    # print(is_hermitian(T_1))
    # print(is_psd(T_1))

    # T_2 = I4 - T_1
    # print(is_hermitian(T_2))
    # print(is_psd(T_2))
    # # print(T_2)
    # dT = T_1 - T_2
    # print(is_hermitian(dT))
    # print(is_psd(dT))
    # print(dT)

    # print(get_psd_herm_neig(np.random.normal(scale=10, size=16), 4))
    pass

if __name__ == '__main__':
    __tests__()