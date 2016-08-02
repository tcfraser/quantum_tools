import random
from ..utilities.utils import *

def u2pi():
    return random.uniform(0.0,2 * np.pi)

def upi():
    return random.uniform(0.0,np.pi)

def cholesky_T(t):
    # James, Kwiat, Munro, and White Physical Review A 64 052312
    # T = np.array([
    #     [      t[0]      ,        0        ,       0         ,   0   ],
    #     [  t[1] + i*t[2] ,      t[3]       ,       0         ,   0   ],
    #     [  t[4] + i*t[5] ,  t[6] + i*t[7]  ,      t[8]       ,   0   ],
    #     [ t[9] + i*t[10] , t[11] + i*t[12] , t[13] + i*t[14] , t[15] ],
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
    return T


def uniform_n_sphere_metric(n):
    """
        for n = 3: returns
        (cos x, sin x cos y, sin x sin y)
        for n = 4: returns
        (cos x, sin x cos y, sin x sin y cos z, sin x sin y sin z)
    """
    angles = [u2pi() for _ in range(n-1)]
    trig_pairs = [(math.cos(angles[i]), math.sin(angles[i])) for i in range(n-1)]
    # trig_pairs = [(math.cos(u2pi()), math.sin(u2pi())) for i in range(n-1)]
    metric = np.ones(n)
    # metric = ['', '', '', '']
    for i in range(n):
        for j in range(i+1):
            if j < n-1:
                if i == j:
                    metric[i] *= trig_pairs[j][0] # cos
                    # metric[i] += 'cos({0})'.format(j) # cos
                else:
                    metric[i] *= trig_pairs[j][1] # sin
                    # metric[i] += 'sin({0})'.format(j) # sin
        # print(metric[i])
    # print(metric)
    # print(linalg.norm(metric))
    # print(trig_pairs)
    assert(is_close(linalg.norm(metric), 1.0)), "Not normalized."
    return metric


def cholesky(t):
    T = cholesky_T(t)
    Td = T.conj().T
    g = np.dot(Td, T)
    # assert(is_hermitian(g)), "g not hermitian!"
    # assert(is_psd(g)), "g not positive semi-definite!"
    return g

def param_GL_C(t):
    assert(is_square(len(t)/2)), "Number of parameters needs to be a twice a square number, not {0}.".format(len(t))
    size = int((len(t)/2)**(1/2))
    t = np.asarray(t)
    GL_RR = np.reshape(t, (2, size, size))
    GL_C = GL_RR[0] + i * GL_RR[1]
    return GL_C


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