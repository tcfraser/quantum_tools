import numpy as np
from cmath import pi, exp
from itertools import product

def real():
    return np.random.normal()

def complex():
    return real() + 1j * real()

def GL_R(n):
    return np.random.normal(size=((n,n)))

def GL_C(n):
    return np.random.normal(size=((n,n))) \
    + 1j * np.random.normal(size=((n,n)))

def clock(n):
    w = exp(2j*pi/n)
    clock_matrix = np.zeros((n, n), dtype='complex')
    for k in range(n):
        clock_matrix[k, k] = w**k
    return clock_matrix

def shift(n):
    shift_matrix = np.zeros((n,n), dtype='complex')
    for k in range(n):
        shift_matrix[(k + 1) % n, k] = 1
    return shift_matrix

def pauli(n):
    c, s = clock(n), shift(n)
    return [np.dot(np.linalg.matrix_power(c, i),np.linalg.matrix_power(s, j))  for i, j in product(range(n),range(n))]

def GL_knit_QR(GL_n):
    Q, R = np.linalg.qr(GL_n)
    r = np.diagonal(R)
    if np.any(r == 0):
        raise Exception("Singular matrix.")
    L = np.diag(r/np.abs(r))
    return np.dot(Q, L)

def U(n):
    return GL_knit_QR(GL_C(n))

def P_I(dim, count):
    S = [np.zeros((dim,dim)) for _ in range(count)]
    for i in range(dim):
        S[np.random.randint(0, count)][i,i] = 1
    return S

def perform_tests():
    # print(random_real())
    # print(complex())
    # print(GL_C(4))
    # print(U(4))
    print(P_I(4))


if __name__ == '__main__':
    perform_tests()