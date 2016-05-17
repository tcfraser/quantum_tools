import itertools
import numpy as np
from scipy import linalg
from functools import reduce

# === Constants ===
i = 1j
mach_tol = 1e-12
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

# === Utils ===
def gen_memory_slots(mem_loc):
    i = 0
    slots = []
    for m_size in mem_loc:
        slots.append(np.arange(i, i+m_size))
        i += m_size
    return slots

def normalize(a):
    a /= linalg.norm(a)

def norm_real_parameter(x):
    return np.cos(x)**2

def ket_to_dm(ket):
    return np.outer(ket, ket.conj())

def ei(x):
    return np.exp(1j*x)

def tensor(*args):
    return reduce(np.kron, args)

def is_hermitian(A):
    return np.array_equal(A, A.conj().T)

def is_psd(A):
    return np.all(linalg.eigvals(A) >= -mach_tol)

def is_trace_one(A):
    return is_close(np.trace(A), 1)

def is_close(a,b):
    return abs(a - b) < mach_tol

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

def get_psd_herm(t, size, unitary_trace=True):
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
    if unitary_trace:
        g /= np.trace(g)
        # assert(is_trace_one(groundn)), "Trace of g is not 1.0! Difference: {0}".format(np.trace(g) - 1)
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

def __tests__():
    print(get_maximally_entangled_bell_state(0))
    print(get_maximally_entangled_bell_state(1))
    print(get_maximally_entangled_bell_state(2))
    print(get_maximally_entangled_bell_state(3))
    print(get_maximally_entangled_bell_state(4))

if __name__ == '__main__':
    __tests__()